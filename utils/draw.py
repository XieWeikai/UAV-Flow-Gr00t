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


def plot_2d_trajectory_with_yaw(
	poses: np.ndarray,
	save_path: str = "trajectory_vis.png",
	title: str = "Trajectory Visualization (Body Frame)",
	xlabel: str = "X (m)",
	ylabel: str = "Y (m)",
) -> None:
	"""
	Plot 2D trajectory (x, y) with yaw orientation from poses [x, y, z, yaw].
	yaw is expected in degrees.
	"""
	try:
		import matplotlib.pyplot as plt
	except ImportError as e:
		raise RuntimeError("matplotlib is required for plotting") from e

	if poses.ndim != 2 or poses.shape[1] < 4:
		raise ValueError(f"Poses must be of shape [N, >=4] (x, y, z, yaw), got {poses.shape}")

	x = poses[:, 0]
	y = poses[:, 1]
	z = poses[:, 2]
	yaw = poses[:, 3]

	plt.figure(figsize=(10, 8))
	
	# Plot trajectory with Z as color
	# Connect points with a thin line to visualize the sequence
	plt.plot(x, y, color='gray', alpha=0.3, linewidth=1, zorder=1)
	# Scatter points colored by Z height
	sc = plt.scatter(x, y, c=z, cmap='viridis', label='Trajectory (Color=Z)', s=10, alpha=0.8, zorder=2)
	plt.colorbar(sc, label='Z (Height in m)')

	# Plot start and end
	plt.scatter(x[0], y[0], color='green', label='Start', s=100, zorder=5, edgecolors='black')
	plt.scatter(x[-1], y[-1], color='red', label='End', s=100, zorder=5, edgecolors='black')

	# Quiver plot for orientation (yaw)
	yaw_rad = np.deg2rad(yaw)
	u = np.cos(yaw_rad)
	v = np.sin(yaw_rad)

	step = max(1, len(x) // 20)
	plt.quiver(x[::step], y[::step], u[::step], v[::step], angles='xy', scale_units='xy', scale=2, color='orange', alpha=0.8, label='Orientation', width=0.003)

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.axis('equal')
	plt.legend()
	plt.grid(True)

	plt.savefig(save_path)
	print(f"Saved visualization to {save_path}")
	plt.close()


def animate_trajectory_with_goals(
	poses: np.ndarray,
	goal_frame_idx: np.ndarray,
	save_path: str = "trajectory_goals.gif",
	title: str = "Trajectory Goal Visualization",
	xlabel: str = "X (m)",
	ylabel: str = "Y (m)",
	fps: int = 5,
) -> None:
	"""
	Create an animation showing the agent moving along the trajectory and its current goal.
	
	poses: [N, 4] array of [x, y, z, yaw] (yaw in degrees)
	goal_frame_idx: [N] array of relative goal frame indices. -1 means no goal.
	save_path: Output file path (.gif or .mp4)
	"""
	try:
		import matplotlib.pyplot as plt
		from matplotlib.animation import FuncAnimation, PillowWriter
	except ImportError as e:
		raise RuntimeError("matplotlib is required for plotting") from e

	if poses.ndim != 2 or poses.shape[1] < 4:
		raise ValueError(f"Poses must be of shape [N, >=4] (x, y, z, yaw), got {poses.shape}")
	
	N = len(poses)
	if len(goal_frame_idx) != N:
		raise ValueError(f"Length of goal_frame_idx ({len(goal_frame_idx)}) must match poses ({N})")

	x = poses[:, 0]
	y = poses[:, 1]
	yaw = poses[:, 3]

	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Background trajectory
	ax.plot(x, y, color='gray', alpha=0.3, linewidth=1, label='Full Path')
	ax.scatter(x[0], y[0], color='green', label='Start', s=50, edgecolors='black', zorder=2)
	ax.scatter(x[-1], y[-1], color='red', label='End', s=50, edgecolors='black', zorder=2)
	
	# Elements to update in animation
	# Z-order: 
	# 1: Gray Trajectory Line
	# 2: Start/End Markers
	# 3: Connection Line
	# 4: No Goal Markers (Scatter & Quiver)
	# 5: Current Agent (Scatter & Quiver)
	# 6: Goal Marker
	
	current_point, = ax.plot([], [], 'bo', markersize=8, label='Current Agent', zorder=5)
	goal_point, = ax.plot([], [], 'mx', markersize=10, markeredgewidth=3, label='Goal', zorder=6)
	connection_line, = ax.plot([], [], 'b--', alpha=0.5, linewidth=1, zorder=3)
	
	# Persistent no-goal points
	# 'sienna' is a coffee-like brown
	no_goal_scatter = ax.scatter([], [], c='sienna', s=40, label='No Goal', zorder=4, marker='o', edgecolors='k') 
	no_goal_quiver = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=2, color='sienna', width=0.005, zorder=4)

	# Quiver for orientation (dynamic)
	# Initializing with a default point to ensure properties are set correctly
	quiver = ax.quiver([x[0]], [y[0]], [np.cos(np.deg2rad(yaw[0]))], [np.sin(np.deg2rad(yaw[0]))], angles='xy', scale_units='xy', scale=1, color='blue', width=0.008, zorder=5)

	# Goal orientation quiver (dynamic)
	goal_quiver = ax.quiver([x[0]], [y[0]], [np.cos(np.deg2rad(yaw[0]))], [np.sin(np.deg2rad(yaw[0]))], angles='xy', scale_units='xy', scale=1, color='magenta', width=0.008, zorder=6)

	# Text for info
	info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, zorder=10))

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.axis('equal')
	# Update legend to include new elements
	# We need to create a dummy handle for the quiver in the legend if we want it, but usually scatter is enough
	handles, labels = ax.get_legend_handles_labels()
	# remove duplicates if any
	by_label = dict(zip(labels, handles))
	ax.legend(by_label.values(), by_label.keys(), loc='lower right')
	
	ax.grid(True)

	# Set axis limits
	pad = 1.0
	ax.set_xlim(x.min() - pad, x.max() + pad)
	ax.set_ylim(y.min() - pad, y.max() + pad)

	# Store no-goal data
	no_goal_data = {'x': [], 'y': [], 'u': [], 'v': []}

	def init():
		current_point.set_data([], [])
		goal_point.set_data([], [])
		connection_line.set_data([], [])
		
		# Set to a dummy invisible point for quiver since it was initialized with data
		# This prevents size mismatch errors in some mpl versions
		quiver.set_offsets(np.array([[-1e9, -1e9]]))
		quiver.set_UVC([1], [0])
		
		goal_quiver.set_offsets(np.array([[-1e9, -1e9]]))
		goal_quiver.set_UVC([1], [0])
		
		# no_goal artists were initialized empty
		no_goal_scatter.set_offsets(np.empty((0, 2)))
        # no_goal_quiver is handled by recreation in update, so just leave it empty here or clear
        # But since update recreates it, we don't need to do anything complex here, 
        # just ensuring it matches the initial state (empty)
		
		info_text.set_text('')
		
		no_goal_data['x'] = []
		no_goal_data['y'] = []
		no_goal_data['u'] = []
		no_goal_data['v'] = []
		
		return current_point, goal_point, connection_line, quiver, goal_quiver, info_text, no_goal_scatter, no_goal_quiver

	def update(frame):
		nonlocal no_goal_quiver

		# Current pose
		cx, cy = x[frame], y[frame]
		current_point.set_data([cx], [cy])
		
		# Orientation
		c_yaw_rad = np.deg2rad(yaw[frame])
		u = np.cos(c_yaw_rad)
		v = np.sin(c_yaw_rad)
		
		quiver.set_offsets(np.array([[cx, cy]]))
		quiver.set_UVC(np.array([u]), np.array([v]))

		# Goal info
		rel_idx = goal_frame_idx[frame]
		gx, gy = None, None
		
		txt = f"Frame: {frame}/{N}\n"
		
		# Handle Goal/No-Goal
		if rel_idx != -1:
			target_idx = frame + rel_idx
			if 0 <= target_idx < N:
				gx, gy = x[target_idx], y[target_idx]
				g_yaw_rad = np.deg2rad(yaw[target_idx])
				gu = np.cos(g_yaw_rad)
				gv = np.sin(g_yaw_rad)

				goal_point.set_data([gx], [gy])
				connection_line.set_data([cx, gx], [cy, gy])
				
				goal_quiver.set_offsets(np.array([[gx, gy]]))
				goal_quiver.set_UVC(np.array([gu]), np.array([gv]))
				
				txt += f"Goal Rel Idx: {rel_idx}\nTarget Frame: {target_idx}"
			else:
				goal_point.set_data([], [])
				connection_line.set_data([], [])
				# Moves goal quiver out of view
				goal_quiver.set_offsets(np.array([[-1e9, -1e9]]))
				txt += f"Goal Rel Idx: {rel_idx}\n(Out of bounds)"
		else:
			goal_point.set_data([], [])
			connection_line.set_data([], [])
			# Moves goal quiver out of view
			goal_quiver.set_offsets(np.array([[-1e9, -1e9]]))
			txt += "Goal: None (-1)"
			
			# Add to persistent no-goal list
			no_goal_data['x'].append(cx)
			no_goal_data['y'].append(cy)
			no_goal_data['u'].append(u)
			no_goal_data['v'].append(v)
			
			# Update persistent artists
			# Scatter supports resizing via set_offsets
			pts = np.column_stack((no_goal_data['x'], no_goal_data['y']))
			no_goal_scatter.set_offsets(pts)
			
			# Quiver does NOT support resizing well (set_UVC mismatch error).
			# We must recreate it.
			no_goal_quiver.remove()
			no_goal_quiver = ax.quiver(no_goal_data['x'], no_goal_data['y'], 
                                       no_goal_data['u'], no_goal_data['v'], 
                                       angles='xy', scale_units='xy', scale=2, 
                                       color='sienna', width=0.005, zorder=4)
		
		info_text.set_text(txt)
		
		return current_point, goal_point, connection_line, quiver, goal_quiver, info_text, no_goal_scatter, no_goal_quiver

	ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=False, interval=100)
	
	if save_path.endswith('.gif'):
		writer = PillowWriter(fps=10)
		ani.save(save_path, writer=writer)
	elif save_path.endswith('.mp4'):
		# Try to use ffmpeg if available
		try:
			ani.save(save_path, writer='ffmpeg', fps=10)
		except Exception:
			print("ffmpeg not found or failed, falling back to gif with .mp4 extension logic (might fail)")
			ani.save(save_path, fps=10)
	else:
		print(f"Unknown extension for {save_path}, saving as gif")
		ani.save(save_path + ".gif", writer=PillowWriter(fps=fps))

	print(f"Saved animation to {save_path}")
	plt.close()


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
