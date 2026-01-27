import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Ensure we can import utils
sys.path.append(os.getcwd())
try:
    from utils.draw import plot_2d_trajectory_with_yaw
except ImportError:
    # If run from test/ dir, we might need to adjust path
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from utils.draw import plot_2d_trajectory_with_yaw


def animate_trajectory_with_goal_coords(
    poses: np.ndarray,
    goal_coords: np.ndarray, # [N, 2] (gx, gy)
    goal_yaws: np.ndarray,   # [N] (gyaw_deg) - absolute global yaw
    save_path: str = "trajectory_goals.gif",
    title: str = "Trajectory Goal Visualization",
):
    """
    poses: [N, 4] (x, y, z, yaw)
    goal_coords: [N, 2] (gx, gy) global coords
    goal_yaws: [N] (global yaw in deg)
    """
    print(f"Creating animation: {save_path}")
    
    x = poses[:, 0]
    y = poses[:, 1]
    yaw = poses[:, 3]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    padding = 2.0
    all_x = np.concatenate([x, goal_coords[:, 0]])
    all_y = np.concatenate([y, goal_coords[:, 1]])
    
    min_x, max_x = np.min(all_x) - padding, np.max(all_x) + padding
    min_y, max_y = np.min(all_y) - padding, np.max(all_y) + padding
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Background path
    ax.plot(x, y, color='gray', alpha=0.3, linewidth=1, label='Full Path')
    ax.scatter(x[0], y[0], color='green', label='Start', s=100, edgecolors='black', zorder=2)
    ax.scatter(x[-1], y[-1], color='red', label='End', s=100, edgecolors='black', zorder=2)
    
    # Elements to update
    current_point, = ax.plot([], [], 'bo', markersize=8, label='Current Agent', zorder=5)
    goal_point, = ax.plot([], [], 'mx', markersize=10, markeredgewidth=3, label='Goal', zorder=6)
    connection_line, = ax.plot([], [], 'b--', alpha=0.5, linewidth=1, zorder=3)
    
    # Dynamic quivers
    # Initializing with a default point
    quiver = ax.quiver([x[0]], [y[0]], [1], [0], angles='xy', scale_units='xy', scale=2, color='blue', width=0.008, zorder=5)
    goal_quiver = ax.quiver([x[0]], [y[0]], [1], [0], angles='xy', scale_units='xy', scale=2, color='magenta', width=0.008, zorder=6)
    
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, zorder=10))

    ax.set_title(title)
    ax.legend(loc='lower right')
    
    def init():
        current_point.set_data([], [])
        goal_point.set_data([], [])
        connection_line.set_data([], [])
        quiver.set_UVC([0], [0])
        goal_quiver.set_UVC([0], [0])
        info_text.set_text('')
        return current_point, goal_point, connection_line, quiver, goal_quiver, info_text
    
    def update(frame):
        # Current pose
        cx, cy = x[frame], y[frame]
        cyaw = yaw[frame]
        
        current_point.set_data([cx], [cy])
        
        # Agent Orientation
        u = np.cos(np.deg2rad(cyaw))
        v = np.sin(np.deg2rad(cyaw))
        quiver.set_offsets(np.array([[cx, cy]]))
        quiver.set_UVC(np.array([u]), np.array([v]))
        
        # Goal info
        gx, gy = goal_coords[frame]
        gyaw = goal_yaws[frame]
        
        goal_point.set_data([gx], [gy])
        connection_line.set_data([cx, gx], [cy, gy])
        
        # Goal Orientation
        gu = np.cos(np.deg2rad(gyaw))
        gv = np.sin(np.deg2rad(gyaw))
        goal_quiver.set_offsets(np.array([[gx, gy]]))
        goal_quiver.set_UVC(np.array([gu]), np.array([gv]))
        
        # Text Info
        dist = np.sqrt((gx-cx)**2 + (gy-cy)**2)
        yaw_diff = (gyaw - cyaw + 180) % 360 - 180
        
        info_text.set_text(
            f"Frame: {frame}\n"
            f"Pos: ({cx:.2f}, {cy:.2f})\n"
            f"Goal: ({gx:.2f}, {gy:.2f})\n"
            f"Dist: {dist:.2f}m\n"
            f"Yaw: {cyaw:.1f}° -> {gyaw:.1f}°\n"
            f"ΔYaw: {yaw_diff:.1f}°"
        )
        
        return current_point, goal_point, connection_line, quiver, goal_quiver, info_text

    anim = FuncAnimation(fig, update, frames=len(poses), init_func=init, blit=False, interval=100)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet_file', type=str)
    parser.add_argument('--output_dir', type=str, default='debug_output')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading {args.parquet_file}...")
    try:
        df = pd.read_parquet(args.parquet_file)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return
    
    if 'observation.state' not in df.columns:
        print("Error: 'observation.state' not found in columns:", df.columns)
        return
        
    # Handle list-columns in parquet
    try:
        poses_list = df['observation.state'].tolist()
        poses = np.array(poses_list) # [N, 4]
        
        actions_list = df['action'].tolist()
        actions = np.array(actions_list) # [N, 8]
    except Exception as e:
        print(f"Error converting columns to numpy: {e}")
        return
    
    print(f"Loaded {len(poses)} frames.")
    
    # Calculate global goals
    goal_coords = []
    goal_yaws = []
    
    for i in range(len(poses)):
        # pose: [x, y, z, yaw]
        curr_x, curr_y, _, curr_yaw = poses[i]
        
        # action goal (relative to body): [gx, gy, gz, gyaw]
        # gx is forward, gy is left in body frame
        rel_gx, rel_gy, _, rel_gyaw = actions[i, 4:]
        
        # Rotation from body to global
        # Body: x-forward, y-left. 
        # Global: standard 2D rotation by yaw (assuming counter-clockwise positive, 0 is East/x)
        # Check coordinate system. If get_poses uses standard math convention:
        yaw_rad = np.deg2rad(curr_yaw)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Apply rotation
        # global_dx = rel_gx * cos(yaw) - rel_gy * sin(yaw)
        # global_dy = rel_gx * sin(yaw) + rel_gy * cos(yaw)
        global_dx = rel_gx * c - rel_gy * s
        global_dy = rel_gx * s + rel_gy * c
        
        abs_gx = curr_x + global_dx
        abs_gy = curr_y + global_dy
        
        # Absolute yaw
        # relative yaw is additive
        abs_gyaw = curr_yaw + rel_gyaw
        
        goal_coords.append([abs_gx, abs_gy])
        goal_yaws.append(abs_gyaw)

    goal_coords = np.array(goal_coords)
    goal_yaws = np.array(goal_yaws)
    
    # Static Plot
    basename = os.path.basename(args.parquet_file).replace('.parquet', '')
    static_path = os.path.join(args.output_dir, f"{basename}_lerobot.png")
    plot_2d_trajectory_with_yaw(poses, save_path=static_path, title=f"{basename} (LeRobot)")
    
    # Animation
    anim_path = os.path.join(args.output_dir, f"{basename}_lerobot.gif")
    animate_trajectory_with_goal_coords(
        poses, 
        goal_coords, 
        goal_yaws, 
        save_path=anim_path, 
        title=f"{basename}"
    )
    
    # Print actions
    print("\n=== First 20 Actions ===")
    np.set_printoptions(precision=3, suppress=True)
    # Print header
    print(f"{'dx':>8} {'dy':>8} {'dz':>8} {'dyaw':>8} | {'gx':>8} {'gy':>8} {'gz':>8} {'gyaw':>8}")
    for i in range(min(20, len(actions))):
        a = actions[i]
        print(f"{a[0]:8.3f} {a[1]:8.3f} {a[2]:8.3f} {a[3]:8.3f} | {a[4]:8.3f} {a[5]:8.3f} {a[6]:8.3f} {a[7]:8.3f}")
    
    print(f"\nResult saved to {static_path} and {anim_path}")


if __name__ == "__main__":
    main()
