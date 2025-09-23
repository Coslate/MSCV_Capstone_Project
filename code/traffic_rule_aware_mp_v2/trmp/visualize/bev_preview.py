from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

def plot_bev(sample: Dict[str, Any], out_png: str = None, show: bool = False):
    """Simple BEV visualization:
    - plots LiDAR xy (if provided as Nx4)
    - plots agents' past xy trajectories
    - (optionally) draws some map polylines if provided as explicit coords (this skeleton often holds tokens/ids only)
    """
    plt.figure(figsize=(6,6))
    ax = plt.gca(); ax.set_aspect('equal'); ax.grid(True, linestyle=':')

    # LiDAR
    if isinstance(sample.get("lidar"), np.ndarray):
        pts = sample["lidar"]
        if pts.shape[1] >= 2:
            ax.scatter(pts[:,0], pts[:,1], s=0.2, alpha=0.5, label="LiDAR")

    # Agents
    for aid, rec in sample.get("agents_history", {}).items():
        xy = rec.get("xy")
        if isinstance(xy, np.ndarray) and xy.shape[0] > 0:
            ax.plot(xy[:,0], xy[:,1], linewidth=1.0, alpha=0.9)

    # Ego history (world coordinate)
    eh = sample.get("ego_history")
    if isinstance(eh, np.ndarray) and eh.size:
        ax.plot(eh[:,0], eh[:,1], linestyle='--', linewidth=2.0, alpha=0.9, label="ego")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title("BEV Preview")

    if out_png: plt.savefig(out_png, dpi=150, bbox_inches='tight')
    if show: plt.show()
    plt.close()
