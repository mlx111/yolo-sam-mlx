from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    vertex_count = None
    has_color = False
    header_end = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex "):
            vertex_count = int(stripped.split()[-1])
        elif stripped == "property uchar red":
            has_color = True
        elif stripped == "end_header":
            header_end = index + 1
            break

    if vertex_count is None or header_end is None:
        raise ValueError(f"Invalid PLY header: {path}")

    data_lines = lines[header_end : header_end + vertex_count]
    data = np.loadtxt(data_lines, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    points = data[:, :3]
    colors = None
    if has_color and data.shape[1] >= 6:
        colors = np.clip(data[:, 3:6] / 255.0, 0.0, 1.0)
    return points, colors


def preview_pointcloud(ply_path: Path, output_path: Path, max_points: int = 30000) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points, colors = read_ascii_ply(ply_path)
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
        colors = colors[::step] if colors is not None else None

    fig = plt.figure(figsize=(8, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors if colors is not None else "tab:blue",
        s=1.5,
        depthshade=False,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(ply_path.name)
    ax.view_init(elev=25, azim=-55)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an ASCII PLY point cloud to a PNG preview without WebGL.")
    parser.add_argument("ply", help="Input ASCII PLY path.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--max-points", type=int, default=30000)
    args = parser.parse_args()

    ply_path = Path(args.ply).resolve()
    output_path = Path(args.out).resolve() if args.out else ply_path.with_suffix(".preview.png")
    print(preview_pointcloud(ply_path, output_path, max_points=args.max_points))


if __name__ == "__main__":
    main()

