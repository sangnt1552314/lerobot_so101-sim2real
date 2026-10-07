"""Automatically align the simulated base_camera with the real third-view camera.

The real robot is moved between its rest pose and a few poses rotated about the base (shoulder_pan only,
so the folded arm stays clear of the table). Differencing the real images taken at those poses gives the
pixels where the robot changed, without needing a segmentation model. The sim camera position, look-at
target and fov are then searched so that the same change, rendered in sim, covers the same pixels (soft
IoU of the pairwise change masks), first with a broad random search and then refined with Nelder-Mead.
Two extra poses only raise the wrist, which isolates where the gripper really is; the final refinement also
fits constant offsets on shoulder_lift / elbow_flex / wrist_flex, which absorbs (and reports) calibration
errors that would otherwise drag the camera estimate off.

    # no robot needed: checks the pipeline recovers a known, perturbed camera
    python lerobot_sim2real/scripts/auto_camera_alignment.py --self-test

    # real alignment; --write updates env_config.json (a .bak copy is kept)
    python lerobot_sim2real/scripts/auto_camera_alignment.py --write

    # re-run the search on frames captured earlier, without moving the arm
    python lerobot_sim2real/scripts/auto_camera_alignment.py --reuse-captures camera_alignment
"""

from dataclasses import dataclass, field
import json
import os
import shutil
import time
from typing import List, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import tyro
from scipy.optimize import minimize

import mani_skill.envs  # noqa: F401
from mani_skill.utils import sapien_utils


@dataclass
class Args:
    env_id: str = "SO101GraspCube-v1"
    env_kwargs_json_path: str = "env_config.json"
    """env config holding the current base_camera_settings, used as the starting guess"""
    robot_uid: str = "so101"
    robot_port: Optional[str] = None
    """serial port override, defaults to lerobot_sim2real/config/real_robot.py"""
    robot_id: Optional[str] = None
    """calibration id override, defaults to lerobot_sim2real/config/real_robot.py"""
    pan_offsets: List[float] = field(default_factory=lambda: [0.4, -0.4, 0.8, -0.8])
    """shoulder_pan offsets (rad) from the rest pose used to make the robot move in the image"""
    wrist_offsets: List[float] = field(default_factory=lambda: [-0.4, -0.8])
    """wrist_flex offsets (rad) from the rest pose; negative raises the gripper away from the table"""
    fit_joint_offsets: bool = True
    """also fit constant calibration offsets for shoulder_lift, elbow_flex and wrist_flex"""
    res: int = 192
    """square resolution the real (center cropped) and sim images are compared at"""
    diff_threshold: float = 25.0
    """per-pixel color change (0-255) counted as robot motion in the real images"""
    num_random: int = 4000
    """number of random camera candidates in the broad search"""
    num_refine: int = 8
    """number of best random candidates refined with Nelder-Mead"""
    out_dir: str = "camera_alignment"
    """where the before/after overlay image and captured frames are saved"""
    write: bool = False
    """write the result into env_kwargs_json_path (keeps a .bak copy)"""
    reuse_captures: Optional[str] = None
    """directory with real_pose{i}.png (and poses.json) from an earlier run; skips moving the robot"""
    self_test: bool = False
    """do not touch the robot; render "real" images from a known perturbed camera and check it is recovered"""
    seed: int = 0


def center_crop_resize(img: np.ndarray, res: int) -> np.ndarray:
    """same preprocessing Sim2RealEnv applies to real camera images"""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0 : y0 + s, x0 : x0 + s], (res, res), interpolation=cv2.INTER_AREA)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask.astype(bool)


def match_colors(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """per-channel affine match of src to ref (20th/80th percentiles), undoing auto exposure / white balance
    changes between captures. The robot covers a minority of pixels so the percentiles are mostly background."""
    out = np.empty_like(src, dtype=np.float32)
    for c in range(3):
        s_lo, s_hi = np.percentile(src[..., c], [20, 80])
        r_lo, r_hi = np.percentile(ref[..., c], [20, 80])
        out[..., c] = (src[..., c] - s_lo) * (r_hi - r_lo) / max(s_hi - s_lo, 1e-3) + r_lo
    return out


def motion_mask(img_a: np.ndarray, img_b: np.ndarray, threshold: float) -> np.ndarray:
    """real side: pixels whose color changed between two robot poses. The threshold adapts to the image-wide
    noise level (3x the median difference) since cheap webcams can shift a whole frame by ~20 levels."""
    a = cv2.GaussianBlur(img_a.astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(match_colors(img_b.astype(np.float32), img_a.astype(np.float32)), (5, 5), 0)
    diff = np.abs(a - b).max(axis=-1)
    return clean_mask(diff > max(threshold, 3 * float(np.median(diff))))


def change_mask(seg_a: np.ndarray, seg_b: np.ndarray, robot_ids: np.ndarray) -> np.ndarray:
    """sim side, modelling what image differencing sees: pixels where the visible object changes and the
    robot is involved. Unlike robot_a XOR robot_b this also counts one robot link replacing another."""
    robot_a, robot_b = np.isin(seg_a, robot_ids), np.isin(seg_b, robot_ids)
    return clean_mask((seg_a != seg_b) & (robot_a | robot_b))


def soften(mask: np.ndarray) -> np.ndarray:
    # blurred masks give the optimizer a smoother objective than hard pixel IoU
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)


def soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-6))


class SimCamera:
    """renders robot-only masks for a given camera (pos, target, fov) and robot qpos"""

    def __init__(self, args: Args, env_kwargs: dict):
        env_kwargs = dict(env_kwargs)
        env_kwargs.pop("greenscreen_overlay_path", None)
        self.env = gym.make(
            args.env_id,
            obs_mode="rgb+segmentation",
            render_mode="sensors",
            domain_randomization=False,
            reward_mode="none",
            sensor_configs=dict(width=args.res, height=args.res),
            **env_kwargs,
        )
        self.env.reset(seed=args.seed)
        self.u = self.env.unwrapped
        self.robot_ids = torch.cat([link.per_scene_id for link in self.u.agent.robot.links]).numpy()
        self.camera = self.u._sensors["base_camera"].camera

    def set_camera(self, params: np.ndarray):
        pos, target, fov = params[:3], params[3:6], float(params[6])
        self.u.camera_mount.set_pose(sapien_utils.look_at(pos.tolist(), target.tolist()))
        self.camera.set_fovy(fov)

    def render(self, params: np.ndarray, qpos: torch.Tensor):
        """returns (per-pixel actor ids, rgb)"""
        self.set_camera(params)
        self.u.agent.robot.set_qpos(qpos)
        obs = self.u._get_obs_sensor_data()["base_camera"]
        return obs["segmentation"][0, ..., 0].cpu().numpy(), obs["rgb"][0].cpu().numpy()

    def change_masks(self, params: np.ndarray, qposes: List[torch.Tensor]) -> List[np.ndarray]:
        segs = [self.render(params, q)[0] for q in qposes]
        return [change_mask(segs[0], segs[k], self.robot_ids) for k in range(1, len(segs))]


def project(params: np.ndarray, pts: np.ndarray, res: int = 128) -> np.ndarray:
    """pixel coordinates of world points for a look-at camera (sapien convention: x forward, y left, z up)"""
    pos, target, fov = params[:3], params[3:6], params[6]
    fwd = (target - pos) / np.linalg.norm(target - pos)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    d = pts - pos
    x, y, z = d @ fwd, d @ left, d @ up
    f = (res / 2) / np.tan(fov / 2)
    return np.stack([res / 2 - f * y / x, res / 2 - f * z / x], -1)


def params_from_settings(s: dict) -> np.ndarray:
    return np.array([*s["pos"], *s["target"], s["fov"]], dtype=np.float64)


def capture_real(args: Args, poses: List[torch.Tensor], sim: SimCamera):
    from lerobot_sim2real.config.real_robot import create_real_robot
    from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent

    robot = create_real_robot(uid=args.robot_uid, port=args.robot_port, robot_id=args.robot_id)
    robot.connect()
    agent = LeRobotRealAgent(robot)
    agent.get_qpos()  # reads motor names needed before sending targets
    camera = robot.cameras["base_camera"]

    print(f"\nThe arm will move slowly to the rest pose, then rotate about its base by {args.pan_offsets} rad.")
    input("Make sure the area around the arm is clear, then press Enter (Ctrl+C to abort)... ")
    images, qposes = [], []
    try:
        for i, q in enumerate(poses):
            agent.reset(q)
            time.sleep(1.5)  # let the arm settle and the camera auto-exposure adapt
            for _ in range(5):
                camera.async_read()  # drop stale frames
                time.sleep(0.05)
            img = camera.async_read()
            images.append(center_crop_resize(img, args.res))
            qposes.append(agent.get_qpos().flatten().clone())
            cv2.imwrite(os.path.join(args.out_dir, f"real_pose{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"captured pose {i}: qpos={np.round(qposes[-1].numpy(), 3).tolist()}")
        with open(os.path.join(args.out_dir, "poses.json"), "w") as f:
            json.dump([q.tolist() for q in qposes], f)
    finally:
        print("returning arm to the folded rest keyframe")
        agent.reset(torch.tensor(sim.u.agent.keyframes["rest"].qpos, dtype=torch.float32))
        robot.disconnect()
    return images, qposes


def load_captures(args: Args, poses: List[torch.Tensor]):
    images = []
    for i in range(len(poses)):
        img = cv2.imread(os.path.join(args.reuse_captures, f"real_pose{i}.png"))
        if img is None:
            raise FileNotFoundError(f"{args.reuse_captures}/real_pose{i}.png (pan_offsets / wrist_offsets must match the earlier run)")
        images.append(center_crop_resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), args.res))
    poses_path = os.path.join(args.reuse_captures, "poses.json")
    if os.path.exists(poses_path):
        with open(poses_path) as f:
            qposes = [torch.tensor(q, dtype=torch.float32) for q in json.load(f)]
    else:
        print("no poses.json found, using the commanded poses instead of measured ones")
        qposes = poses
    return images, qposes


def main(args: Args):
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with open(args.env_kwargs_json_path) as f:
        env_kwargs = json.load(f)
    init_params = params_from_settings(env_kwargs["base_camera_settings"])

    sim = SimCamera(args, env_kwargs)
    rest = sim.u.rest_qpos.cpu().float().flatten()
    poses = [rest.clone()]
    for off in args.pan_offsets:
        q = rest.clone()
        q[0] += off
        poses.append(q)
    for off in args.wrist_offsets:
        q = rest.clone()
        q[3] += off
        poses.append(q)

    if args.self_test:
        true_params = init_params + np.array([0.08, -0.06, 0.05, 0.03, 0.02, 0.0, 0.08])
        images = [sim.render(true_params, q)[1] for q in poses]
        qposes = poses
        print(f"self-test: hidden camera = {np.round(true_params, 4).tolist()}")
    elif args.reuse_captures is not None:
        images, qposes = load_captures(args, poses)
    else:
        images, qposes = capture_real(args, poses, sim)

    real_masks = [motion_mask(images[0], images[k], args.diff_threshold) for k in range(1, len(images))]
    coverage = [m.mean() for m in real_masks]
    print(f"real motion mask coverage per pose pair: {np.round(coverage, 3).tolist()}")
    if min(coverage) < 0.002:
        raise RuntimeError(
            "The robot barely moves in the camera image. Check the camera faces the robot and the right "
            f"camera index is configured; frames are saved in {args.out_dir}/"
        )
    real_soft = [soften(m) for m in real_masks]

    def with_offsets(offsets: np.ndarray) -> List[torch.Tensor]:
        """measured qposes with constant offsets added to shoulder_lift, elbow_flex, wrist_flex"""
        out = []
        for q in qposes:
            q = q.clone()
            q[1:4] += torch.as_tensor(offsets, dtype=q.dtype)
            out.append(q)
        return out

    def score(x: np.ndarray) -> float:
        """x = camera params (7), optionally followed by the 3 joint offsets"""
        params, offsets = x[:7], (x[7:] if len(x) > 7 else np.zeros(3))
        if params[6] < 0.2 or params[6] > 2.5 or np.abs(offsets).max() > 0.35:
            return 0.0
        masks = sim.change_masks(params, with_offsets(offsets))
        return float(np.mean([soft_iou(soften(m), r) for m, r in zip(masks, real_soft)]))

    init_score = score(init_params)
    print(f"starting camera {np.round(init_params, 4).tolist()} soft IoU={init_score:.3f}")

    # broad search: a quarter of the samples around the starting guess, the rest anywhere around the robot,
    # parameterized as a look-at target near the arm plus direction / elevation / distance from it
    local_scale = np.array([0.15, 0.15, 0.15, 0.08, 0.08, 0.04, 0.2])
    candidates = [init_params]
    t0 = time.time()
    for i in range(args.num_random):
        if i % 4 == 0:
            p = init_params + rng.normal(size=7) * local_scale
        else:
            target = rng.uniform([-0.05, -0.25, 0.0], [0.35, 0.25, 0.25])
            az, el, dist = rng.uniform(-np.pi, np.pi), rng.uniform(-0.1, 1.45), rng.uniform(0.25, 1.0)  # up to ~83 deg down
            pos = target + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            p = np.concatenate([pos, target, [rng.uniform(0.5, 1.4)]])
        if np.linalg.norm(p[:2]) < 0.15 or p[2] < 0.0:  # camera inside the robot or under the table
            continue
        candidates.append(p)
    scored = sorted(((score(p), p) for p in candidates), key=lambda x: -x[0])
    print(f"random search: {len(candidates)} candidates in {time.time() - t0:.0f}s, best soft IoU={scored[0][0]:.3f}")

    refined = []
    for s, p in scored[: args.num_refine]:
        res = minimize(
            lambda x: -score(x), p, method="Nelder-Mead",
            options=dict(maxiter=1000, xatol=1e-4, fatol=1e-4, initial_simplex=np.vstack([p, p + np.diag(local_scale * 0.3)])),
        )
        refined.append((-res.fun, res.x))
        print(f"  refined {s:.3f} -> {-res.fun:.3f}")
    refined.sort(key=lambda t: -t[0])
    best_score, best_params = refined[0]
    best_offsets = np.zeros(3)

    if args.fit_joint_offsets:
        step = np.r_[local_scale * 0.3, [0.05, 0.05, 0.05]]
        for s, p in refined[:3]:
            x0 = np.r_[p, np.zeros(3)]
            res = minimize(
                lambda x: -score(x), x0, method="Nelder-Mead",
                options=dict(maxiter=2500, xatol=1e-4, fatol=1e-4, initial_simplex=np.vstack([x0, x0 + np.diag(step)])),
            )
            print(f"  with joint offsets {s:.3f} -> {-res.fun:.3f}  offsets(deg) {np.round(np.rad2deg(res.x[7:]), 1).tolist()}")
            if -res.fun > best_score:
                best_score, best_params, best_offsets = -res.fun, res.x[:7], res.x[7:]
        qposes = with_offsets(best_offsets)
        offsets_deg = dict(zip(["shoulder_lift", "elbow_flex", "wrist_flex"], np.round(np.rad2deg(best_offsets), 2).tolist()))
        with open(os.path.join(args.out_dir, "joint_offsets.json"), "w") as f:
            json.dump(offsets_deg, f, indent=2)
        print(f"fitted joint offsets (real = reported + offset), degrees: {offsets_deg}")
        if np.abs(np.rad2deg(best_offsets)).max() > 3:
            print("  offsets above ~3 deg suggest recalibrating those joints (or correcting them in LeRobotRealAgent),")
            print("  otherwise the policy sees the arm in a different pose than it believes it is in")

    # hard IoU on the robot masks for reporting
    best_masks = sim.change_masks(best_params, qposes)
    hard_iou = np.mean([(s & r).sum() / max((s | r).sum(), 1) for s, r in zip(best_masks, real_masks)])
    result = dict(
        pos=np.round(best_params[:3], 4).tolist(),
        target=np.round(best_params[3:6], 4).tolist(),
        fov=round(float(best_params[6]), 4),
    )
    print(f"\nbest camera: {json.dumps(result)}  soft IoU={best_score:.3f}  motion-mask IoU={hard_iou:.3f}")
    # distance and fov trade off (a farther, narrower camera looks almost the same), so judge the result in
    # image space: how far cube spawn area points land from where they should, at the 128x128 policy resolution
    cx, cy = env_kwargs.get("spawn_box_pos", [0.3, 0.05])
    h = env_kwargs.get("spawn_box_half_size", 0.1)
    g = np.meshgrid(np.linspace(cx - h, cx + h, 9), np.linspace(cy - h, cy + h, 9), [0.02])
    spawn_pts = np.stack([a.ravel() for a in g], -1)
    ref = true_params if args.self_test else best_params
    for name, p in (("start", init_params), ("aligned", best_params)):
        e = np.linalg.norm(project(p, spawn_pts) - project(ref, spawn_pts), axis=-1)
        label = "vs hidden camera" if args.self_test else "shift vs aligned"
        print(f"{name:>8} {label}, cube spawn area @128px: mean {e.mean():.2f} px, max {e.max():.2f} px")

    # overlay: real rest image with the sim robot outline before (red) and after (green) alignment
    overlay = cv2.cvtColor(images[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
    for params, color in ((init_params, (0, 0, 255)), (best_params, (0, 255, 0))):
        m = np.isin(sim.render(params, qposes[0])[0], sim.robot_ids).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, color, 1)
    real_row = np.concatenate([m.astype(np.uint8) * 255 for m in real_masks], axis=1)
    sim_row = np.concatenate([m.astype(np.uint8) * 255 for m in best_masks], axis=1)
    overlay_path = os.path.join(args.out_dir, "alignment_overlay.png")
    masks_path = os.path.join(args.out_dir, "alignment_masks.png")
    cv2.imwrite(overlay_path, cv2.resize(overlay, (512, 512), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(masks_path, np.concatenate([real_row, sim_row], axis=0))
    print(f"saved {overlay_path} (real rest frame, sim robot outline before = red, after = green)")
    print(f"saved {masks_path} (top: real change masks per pose, bottom: aligned sim change masks)")

    if args.write and not args.self_test:
        shutil.copy(args.env_kwargs_json_path, args.env_kwargs_json_path + ".bak")
        env_kwargs["base_camera_settings"].update(result)
        with open(args.env_kwargs_json_path, "w") as f:
            json.dump(env_kwargs, f, indent=2)
        print(f"updated {args.env_kwargs_json_path} (previous version in {args.env_kwargs_json_path}.bak)")
    sim.env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
