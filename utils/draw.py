# Utility to plot 3D trajectories (x, y, z)
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, TYPE_CHECKING
from pathlib import Path

import numpy as np
import io
try:
	# Try to import Pillow at module import time so individual functions
	# don't need to import it. If Pillow is not available, leave
	# _PIL_Image as None and the function will gracefully fall back to
	# returning None.
	from PIL import Image as _PIL_Image
except Exception:  # pragma: no cover - optional dependency
	_PIL_Image = None

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
	from PIL.Image import Image as PILImage


def _as_np_array(points: Iterable) -> np.ndarray:
	arr = np.asarray(points, dtype=float)
	if arr.ndim == 1 and arr.size == 3:
		arr = arr.reshape((1, 3))
	if arr.ndim != 2 or arr.shape[1] != 3:
		raise ValueError("points must be an iterable of (x, y, z) coordinates")
	return arr


def set_axes_equal(ax) -> None:
	"""Set 3D plot axes to equal scale.

	Matplotlib's 3D plots are not equal aspect by default. This helper makes
	x, y and z have the same scale so trajectories don't look skewed.
	"""
	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	x_range = abs(x_limits[1] - x_limits[0])
	x_mid = np.mean(x_limits)
	y_range = abs(y_limits[1] - y_limits[0])
	y_mid = np.mean(y_limits)
	z_range = abs(z_limits[1] - z_limits[0])
	z_mid = np.mean(z_limits)

	# The plot radius is half the maximum range
	plot_radius = 0.5 * max(x_range, y_range, z_range)

	ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
	ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
	ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def plot_3d_trajectory(
	points: Sequence[Sequence[float]] | np.ndarray | dict[str, Sequence[Sequence[float]]],
	save_path: Optional[str | Path] = None,
	show: bool = False,
	title: Optional[str] = None,
	color: str = "C0",
	linewidth: float = 2.0,
	marker: Optional[str] = "o",
	markersize: float = 3.0,
	elev: float = 30.0,
	azim: float = 90.0,
	equal_axis: bool = True,
	) -> Optional["PILImage"]:
	"""Plot a 3D trajectory given an iterable of (x, y, z) points.

	Parameters
	- points: sequence-like of shape (N, 3)
	- save_path: if provided, the plot will be saved to this path (png/svg/pdf supported)
	- show: if True, call matplotlib.pyplot.show() (may block)
	- title: optional title string
	- color, linewidth, marker, markersize: visual options
	- elev, azim: view elevation and azimuth
	- equal_axis: whether to force equal scaling on the 3 axes

	The function returns a PIL.Image.Image (RGBA) when Pillow is available
	and the in-memory rendering succeeds, otherwise it returns None.
	The figure is still saved to `save_path` or shown interactively according
	to the provided arguments.
	"""
	try:
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	except Exception as e:  # pragma: no cover - import errors handled at runtime
		raise RuntimeError("matplotlib is required to plot trajectories") from e

	# Accept either a single trajectory (sequence of points) or a dict of named trajectories
	if isinstance(points, dict):
		trajectories = points
	else:
		# Backwards compatibility: single unnamed trajectory
		trajectories = {"trajectory": points}

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection="3d")

	# Choose distinct colors for multiple trajectories while allowing a fixed color override
	# Use pyplot's get_cmap to be compatible with newer matplotlib versions
	cmap = plt.get_cmap("tab10")
	names = list(trajectories.keys())

	for idx, name in enumerate(names):
		pts = _as_np_array(trajectories[name])
		if pts.shape[0] == 0:
			continue

		xs = pts[:, 0]
		ys = pts[:, 1]
		zs = pts[:, 2]

		use_color = color if len(names) == 1 else cmap(idx % cmap.N)

		# Draw the trajectory line
		ax.plot(xs, ys, zs, color=use_color, linewidth=linewidth, label=name)

		# Mark every point (optional marker) and also highlight start/end
		if marker is not None:
			# Scatter all points using the provided marker and markersize.
			# Use markersize**2 for the scatter 's' parameter (area in points^2).
			try:
				all_s = float(markersize) ** 2
			except Exception:
				all_s = markersize
			# Scatter all points (Axes3D.scatter expects positional x,y,z)
			ax.scatter(xs, ys, zs, color=use_color, marker=marker, s=all_s, depthshade=True)
			# Highlight start and end with distinct markers
			ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", marker="^", s=markersize * 10)
			ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="red", marker="s", s=markersize * 10)

	# Adopt an FRD (Forward, Right, Down) convention for viewing and labels
	# - X (forward)
	# - Y (right)
	# - Z (down)
	ax.set_xlabel("X (forward)")
	ax.set_ylabel("Y (right)")
	ax.set_zlabel("Z (down)")
	if title:
		ax.set_title(title)

	ax.view_init(elev=elev, azim=azim)

	if equal_axis:
		set_axes_equal(ax)

	# Make positive Z go visually downward to match the FRD (z-down) viewing convention
	try:
		ax.invert_zaxis()
	except Exception:
		# In case of backend peculiarities, fail gracefully without inversion
		pass

	try:
		ax.invert_yaxis()
	except Exception:
		# In case of backend peculiarities, fail gracefully without inversion
		pass

	ax.legend()

	# Always render the figure into an in-memory PNG and try to return a PIL.Image.
	img = None
	try:
		buf = io.BytesIO()
		# Use the same DPI and bbox as when saving to disk
		fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
		buf.seek(0)
		# Use the module-level Pillow import if available (do not import inside)
		if _PIL_Image is not None:
			try:
				img = _PIL_Image.open(buf).convert("RGBA")
			except Exception:
				# If Pillow is not available or image open fails, fall back to None
				img = None
	finally:
		# If a save path was requested, also write the file to disk
		if save_path is not None:
			out = Path(save_path)
			out.parent.mkdir(parents=True, exist_ok=True)
			# save again to the requested path (fig still valid)
			fig.savefig(out, bbox_inches="tight", dpi=200)

		if show:
			plt.show()
		else:
			# Close the figure to free memory in non-interactive use
			plt.close(fig)

	return img


def _demo_spiral(num_points: int = 400, radius: float = 5.0, turns: float = 3.0) -> np.ndarray:
	"""Generate a 3D spiral trajectory for demo/testing."""
	t = np.linspace(0, 1, num_points)
	theta = 2.0 * math.pi * turns * t
	x = radius * t * np.cos(theta)
	y = radius * t * np.sin(theta)
	z = radius * t
	return np.stack([x, y, z], axis=1)


def _demo_cli():
	"""Small CLI used when running this module directly.

	Usage: python -m utils.draw --output out.png [--show]
		   python -m utils.draw --input points.csv --output out.png
	The CSV must be plain with three columns x,y,z and optional header.
	"""
	import argparse

	parser = argparse.ArgumentParser(description="Plot 3D trajectory from points or demo spiral")
	parser.add_argument("--input", type=str, default="test_trajectory.csv", help="Path to CSV file with x,y,z columns")
	parser.add_argument("--output", type=str, default="trajectory_demo.png", help="Output image path")
	parser.add_argument("--show", action="store_true", help="Show the plot interactively")
	parser.add_argument("--points", type=int, default=400, help="Number of demo points when input is not provided")
	args = parser.parse_args()

	if args.input:
		# Resolve input path: if a relative path is provided, try resolving
		# relative to this module's directory first so that bundled CSVs
		# (like test_trajectory.csv placed next to this file) are found.
		input_path = Path(args.input)
		if not input_path.is_absolute():
			module_dir = Path(__file__).resolve().parent
			candidate = module_dir / input_path
			if candidate.exists():
				input_path = candidate

		try:
			pts = np.loadtxt(str(input_path), delimiter=",")
		except Exception:
			# try whitespace separated
			pts = np.loadtxt(str(input_path))
	else:
		pts = _demo_spiral(num_points=args.points)

	plot_3d_trajectory(pts, save_path=args.output, show=args.show, title="3D Trajectory")
	print(f"Saved trajectory plot to: {args.output}")


if __name__ == "__main__":
	# Simple self-test: generate two demo trajectories and save an example image
	out = Path(__file__).resolve().parent / "_test_trajectory.png"
	t1 = _demo_spiral(200, radius=5.0, turns=3.0)
	t2 = _demo_spiral(200, radius=3.0, turns=1.5) + np.array([1.0, -1.0, 0.8])
	plot_3d_trajectory({"spiral_A": t1, "spiral_B": t2}, save_path=str(out), show=False, title="_test_multi_trajectory")
	print(f"Wrote self-test trajectory image to: {out}")
