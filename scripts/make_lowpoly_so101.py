"""Build a low-poly copy of the SO101 robot for faster GPU camera rendering.

The SO101 visual meshes have ~320k triangles per arm, which dominates render time for the 128x128 policy
camera on H100/H200 (few graphics units). This writes decimated copies of the *visual* meshes to
meshes_lowpoly/ and a so101_lowpoly.urdf that uses them. Collision meshes are untouched, so physics is identical.

Use it by setting SO101_URDF_PATH to the generated urdf (see scripts/train_lift_cube_dr_hopper_fast.pbs).

    python scripts/make_lowpoly_so101.py [--target-faces 3000]
"""
import argparse
import re
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

ROBOT_DIR = Path(__file__).resolve().parents[1] / "ManiSkill/mani_skill/assets/robots/so101"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-faces", type=int, default=3000, help="max triangles per visual mesh")
    args = parser.parse_args()

    urdf = (ROBOT_DIR / "so101.urdf").read_text()
    out_dir = ROBOT_DIR / "meshes_lowpoly"
    out_dir.mkdir(exist_ok=True)

    # only rewrite <mesh filename> inside <visual> blocks
    visual_meshes = set()
    for block in re.findall(r"<visual>.*?</visual>", urdf, flags=re.S):
        visual_meshes.update(re.findall(r'filename="(meshes/[^"]+)"', block))

    before = after = 0
    for rel in sorted(visual_meshes):
        mesh = trimesh.load(ROBOT_DIR / rel, force="mesh", process=True)
        n = len(mesh.faces)
        out = out_dir / (Path(rel).stem + ".stl")
        # one pass often stops short of the target on these CAD meshes, so repeat while it keeps shrinking
        for _ in range(5):
            m = len(mesh.faces)
            if m <= args.target_faces:
                break
            v, f = fast_simplification.simplify(
                np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces), 1 - args.target_faces / m, agg=9
            )
            mesh = trimesh.Trimesh(v, f, process=True)
            if len(mesh.faces) > 0.95 * m:
                break
        mesh.export(out)
        before += n
        after += len(mesh.faces)
        print(f"{rel}: {n} -> {len(mesh.faces)} faces")

    def repl(block):
        return re.sub(
            r'filename="meshes/([^"]+?)\.(stl|obj|STL)"', lambda m: f'filename="meshes_lowpoly/{m.group(1)}.stl"', block.group(0)
        )

    (ROBOT_DIR / "so101_lowpoly.urdf").write_text(re.sub(r"<visual>.*?</visual>", repl, urdf, flags=re.S))
    print(f"unique visual meshes: {before} -> {after} faces; wrote {ROBOT_DIR / 'so101_lowpoly.urdf'}")


if __name__ == "__main__":
    main()
