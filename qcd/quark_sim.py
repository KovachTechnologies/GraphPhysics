# quark_sim.py - Unified Quark Graph-Action Simulator
# Modes: neutron (3 quarks) or deuteron (6 quarks)
# Simulations: linear, color, su3

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from tqdm import tqdm
import colorsys
import json
import argparse
from pathlib import Path
import os

if not os.path.exists("graphs"):
    os.mkdir("graphs")

# ========================== LOAD CONFIG ==========================
def load_config():
    parser = argparse.ArgumentParser(description="Unified Quark Graph-Action Simulator")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--seed", type=int, help="Simulation seed")
    parser.add_argument("--mode", choices=["neutron", "deuteron"], help="Particle mode")
    parser.add_argument("--sim", choices=["linear", "color", "su3"], help="Simulation physics level")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        print(f"Warning: {config_path} not found. Using defaults.")
        cfg = {}

    # CLI overrides
    if args.mode:
        cfg["MODE"] = args.mode
    if args.sim:
        cfg["SIMULATION"] = args.sim
    if args.seed is not None:
        cfg["simulation_seed"] = args.seed

    # Defaults
    defaults = {
        "simulation_seed": 42,
        "MODE": "deuteron",
        "SIMULATION": "su3",
        "DIM": 3,
        "CONFINEMENT_STRENGTH": 12.0,
        "REPULSION_STRENGTH": 2.0,
        "REPULSION_EPS": 0.05,
        "COLOR_SINGLET_STRENGTH": 12.0,
        "LEARNING_RATE": 0.07,
        "MAX_ITER": 2000,
        "N_RANDOM_STARTS": 3,
        "DEFAULT_STATIC_ONLY": False,
        "ANIMATE_LIVE": False,
        "SAVE_MP4": True
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v

    return cfg

cfg = load_config()

np.random.seed(cfg["simulation_seed"])
torch.manual_seed(cfg["simulation_seed"])

MODE = cfg["MODE"].lower()
SIMULATION = cfg["SIMULATION"].lower()

# Auto-set N_QUARKS based on MODE (removed redundancy)
N_QUARKS = 6 if MODE == "deuteron" else 3

DIM = cfg["DIM"]
CONFINEMENT_STRENGTH = cfg["CONFINEMENT_STRENGTH"]
REPULSION_STRENGTH = cfg["REPULSION_STRENGTH"]
REPULSION_EPS = cfg["REPULSION_EPS"]
COLOR_SINGLET_STRENGTH = cfg["COLOR_SINGLET_STRENGTH"]
LEARNING_RATE = cfg["LEARNING_RATE"]
MAX_ITER = cfg["MAX_ITER"]
N_RANDOM_STARTS = cfg["N_RANDOM_STARTS"]
DEFAULT_STATIC_ONLY = cfg["DEFAULT_STATIC_ONLY"]
ANIMATE_LIVE = cfg["ANIMATE_LIVE"]
SAVE_MP4 = cfg["SAVE_MP4"]

print(f"Running → MODE: {MODE} | SIMULATION: {SIMULATION} | Quarks: {N_QUARKS} | Runs: {N_RANDOM_STARTS}")

# ========================== GRAPH SETUP ==========================
def create_quark_graph():
    G = nx.complete_graph(N_QUARKS)
    if MODE == "deuteron":
        labels = ['u', 'u', 'd', 'u', 'd', 'd']          # simplified
    else:
        labels = ['u', 'd', 'd']                         # simplified neutron
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label
    return G

# ========================== SU(3) HELPERS ==========================
GELL_MANN = None
if SIMULATION == "su3":
    def get_gell_mann_matrices():
        lambda_matrices = [
            torch.tensor([[0,1,0],[1,0,0],[0,0,0]], dtype=torch.cfloat),
            torch.tensor([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=torch.cfloat),
            torch.tensor([[1,0,0],[0,-1,0],[0,0,0]], dtype=torch.cfloat),
            torch.tensor([[0,0,1],[0,0,0],[1,0,0]], dtype=torch.cfloat),
            torch.tensor([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=torch.cfloat),
            torch.tensor([[0,0,0],[0,0,1],[0,1,0]], dtype=torch.cfloat),
            torch.tensor([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=torch.cfloat),
            torch.tensor([[1,0,0],[0,1,0],[0,0,-2]], dtype=torch.cfloat) / np.sqrt(3)
        ]
        return [lam / 2 for lam in lambda_matrices]
    GELL_MANN = get_gell_mann_matrices()

def random_color_state():
    c = torch.randn(3, dtype=torch.cfloat)
    return c / torch.norm(c)

def color_singlet_fidelity(colors: torch.Tensor, indices=None) -> torch.Tensor:
    if indices is None:
        indices = list(range(3))
    eps = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],
                        [[0,0,-1],[0,0,0],[1,0,0]],
                        [[0,1,0],[-1,0,0],[0,0,0]]], dtype=torch.cfloat, device=colors.device)
    c1, c2, c3 = [colors[i] for i in indices]
    singlet = sum(eps[i,j,k] * c1[i] * c2[j] * c3[k] 
                  for i in range(3) for j in range(3) for k in range(3))
    return torch.abs(singlet)**2

def color_interaction_factor(colors: torch.Tensor) -> torch.Tensor:
    if GELL_MANN is None or SIMULATION != "su3":
        return torch.zeros(1, device=colors.device)
    factor = torch.zeros(1, dtype=torch.float32, device=colors.device)
    for a in range(8):
        T_a = GELL_MANN[a]
        for i in range(N_QUARKS):
            for j in range(i + 1, N_QUARKS):
                Ti_a = torch.real(colors[i].conj() @ T_a @ colors[i])
                Tj_a = torch.real(colors[j].conj() @ T_a @ colors[j])
                factor += Ti_a * Tj_a
    return factor

# ========================== ENERGY ==========================
def total_energy(positions: torch.Tensor, colors: torch.Tensor = None) -> torch.Tensor:
    E = torch.zeros(1, device=positions.device)

    color_factor = color_interaction_factor(colors) if colors is not None else 0.0

    for i in range(N_QUARKS):
        for j in range(i + 1, N_QUARKS):
            r = torch.norm(positions[i] - positions[j])
            E += CONFINEMENT_STRENGTH * (1.0 + color_factor) * r
            E += REPULSION_STRENGTH / (r + REPULSION_EPS)

    if colors is not None:
        if MODE == "deuteron":
            s_p = color_singlet_fidelity(colors, [0,1,2])
            s_n = color_singlet_fidelity(colors, [3,4,5])
            E += COLOR_SINGLET_STRENGTH * (2.0 - s_p - s_n)
        else:
            s = color_singlet_fidelity(colors)
            E += COLOR_SINGLET_STRENGTH * (1.0 - s)

    return E

# ========================== OPTIMIZATION ==========================
def minimize_run(initial_pos_np: np.ndarray):
    positions = torch.tensor(initial_pos_np, dtype=torch.float32, requires_grad=True)
    
    if SIMULATION == "linear":
        colors = None
        params = [positions]
    else:
        colors = torch.stack([random_color_state() for _ in range(N_QUARKS)], dim=0)
        colors.requires_grad = True
        params = [positions, colors]

    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    
    history_pos = []
    history_colors = []
    energies = []

    for it in tqdm(range(MAX_ITER), desc=f"Optimizing {MODE}-{SIMULATION}"):
        optimizer.zero_grad()
        loss = total_energy(positions, colors)
        loss.backward()
        optimizer.step()

        if colors is not None:
            with torch.no_grad():
                for c in colors:
                    c /= torch.norm(c)

        history_pos.append(positions.detach().clone().numpy())
        if colors is not None:
            history_colors.append(colors.detach().clone().numpy())
        energies.append(loss.item())

        if it > 200 and len(energies) > 1 and abs(energies[-1] - energies[-2]) < 1e-6:
            break

    return np.array(history_pos), (np.array(history_colors) if history_colors else None), np.array(energies)

# ========================== PLOTTING ==========================
def color_vector_to_rgb(c_vec):
    rgb = np.abs(c_vec[:3].real)
    rgb /= (np.sum(rgb) + 1e-8)
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb(h, min(1.0, s*1.4), v)

def plot_3d(G, positions, colors_np, ax, title, energy=None):
    ax.cla()
    
    # Draw nodes and labels
    for i in range(N_QUARKS):
        if colors_np is not None and SIMULATION != "linear":
            rgb = color_vector_to_rgb(colors_np[i])
            if MODE == "deuteron":
                tint = 0.2 if i < 3 else 0.8
                rgb = tuple(0.6 * x + 0.4 * tint for x in rgb)
            ax.scatter(*positions[i], color=rgb, s=380, edgecolor='black', linewidth=2.5, zorder=1)
        else:
            col = ['blue', 'red', 'red'][i % 3]
            ax.scatter(*positions[i], color=col, s=300, edgecolor='black', zorder=1)

        # Improved label placement - offset slightly to avoid overlap
        label = G.nodes[i]['label']
        offset = 0.25
        ax.text(positions[i][0] + offset, 
                positions[i][1] + offset, 
                positions[i][2] + offset,
                f"{label}", 
                color='white', 
                fontsize=10, 
                ha='left', 
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.6), zorder=10)

    # Draw edges
    for i in range(N_QUARKS):
        for j in range(i + 1, N_QUARKS):
            p1, p2 = positions[i], positions[j]
            ax.plot(*zip(p1, p2), color='darkgreen', linewidth=3, alpha=0.85, zorder=2)

    # Singlet info in title
    info = ""
    if colors_np is not None and SIMULATION != "linear":
        if MODE == "deuteron":
            s_p = color_singlet_fidelity(torch.tensor(colors_np, dtype=torch.cfloat), [0,1,2]).item()
            s_n = color_singlet_fidelity(torch.tensor(colors_np, dtype=torch.cfloat), [3,4,5]).item()
            info = f" | p={s_p:.3f} n={s_n:.3f}"
        else:
            s = color_singlet_fidelity(torch.tensor(colors_np, dtype=torch.cfloat)).item()
            info = f" | Singlet={s:.3f}"

    ax.set_title(f"{title}\nE ≈ {energy:.3f}{info}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 13)
    ax.set_ylim(-2, 13)
    ax.set_zlim(-2, 13)


# ========================== MAIN ==========================
if __name__ == "__main__":
    G = create_quark_graph()
    
    histories_pos = []
    histories_colors = []
    final_energies = []
    initial_pos_list = []

    print(f"Starting {N_RANDOM_STARTS} runs — {MODE} + {SIMULATION}")

    for run in range(N_RANDOM_STARTS):
        if MODE == "deuteron":
            init_pos = np.random.rand(N_QUARKS, DIM) * 5 + 1.0
            init_pos[3:] += [6, 0, 0]   # separate clusters
        else:
            init_pos = np.random.rand(N_QUARKS, DIM) * 6 + 1.0

        initial_pos_list.append(init_pos)
        hist_pos, hist_col, energies = minimize_run(init_pos)

        histories_pos.append(hist_pos)
        histories_colors.append(hist_col)
        final_energies.append(energies[-1])

        print(f"Run {run+1:2d} → E = {energies[-1]:.4f}")

    best_idx = int(np.argmin(final_energies))
    best_energy = final_energies[best_idx]
    print(f"\n🎯 BEST RUN: #{best_idx+1} with E = {best_energy:.4f}")

    # Static figure
    if DEFAULT_STATIC_ONLY:
        print("Generating static figure...")
        cols = N_RANDOM_STARTS + 1
        fig = plt.figure(figsize=(6*cols, 9))
        fig.suptitle(f"Quark Simulation — {MODE} | {SIMULATION}", fontsize=16)

        for r in range(N_RANDOM_STARTS):
            ax = fig.add_subplot(2, cols, r+2, projection='3d')
            plot_3d(G, histories_pos[r][-1], 
                    histories_colors[r][-1] if histories_colors[r] is not None else None,
                    ax, f"Run {r+1}", final_energies[r])

        # Initial
        ax_init = fig.add_subplot(2, cols, 1, projection='3d')
        init_col = np.stack([random_color_state().numpy() for _ in range(N_QUARKS)]) if SIMULATION != "linear" else None
        plot_3d(G, initial_pos_list[0], init_col, ax_init, "Initial", 
                total_energy(torch.tensor(initial_pos_list[0]), 
                             torch.stack([random_color_state() for _ in range(N_QUARKS)]) if SIMULATION != "linear" else None).item())

        plt.tight_layout()
        plt.savefig(f"graphs/quark_{MODE}_{SIMULATION}_final.png", dpi=300, bbox_inches='tight')
        plt.show()

    # MP4 of BEST run
    if ANIMATE_LIVE or SAVE_MP4:
        print(f"🎥 Animating BEST run (#{best_idx+1}, E={best_energy:.4f})...")
        hist_pos = histories_pos[best_idx]
        hist_col = histories_colors[best_idx]
        energies = np.array([total_energy(torch.tensor(p), 
                            torch.tensor(c) if c is not None else None).item()
                            for p, c in zip(hist_pos, hist_col if hist_col is not None else [None]*len(hist_pos))])

        fig_anim = plt.figure(figsize=(12, 8))
        ax3d = fig_anim.add_subplot(121, projection='3d')
        ax_e = fig_anim.add_subplot(122)
        energy_line, = ax_e.plot([], [], 'b-', linewidth=2)
        ax_e.set_xlim(0, len(energies))
        ax_e.set_ylim(0, max(energies)*1.05)
        ax_e.set_xlabel("Iteration")
        ax_e.set_ylabel("Graph Action E")
        ax_e.grid(True)

        def animate(frame):
            pos = hist_pos[frame]
            col = hist_col[frame] if hist_col is not None else None
            plot_3d(G, pos, col, ax3d, f"Frame {frame}", energies[frame])
            energy_line.set_data(range(frame+1), energies[:frame+1])
            return ax3d,

        ani = FuncAnimation(fig_anim, animate, frames=len(hist_pos), interval=40, blit=False)

        if ANIMATE_LIVE:
            plt.show()

        if SAVE_MP4:
            try:
                writer = FFMpegWriter(fps=20, metadata=dict(artist='GraphPhysics'), bitrate=2000)
                fname = f"graphs/quark_{MODE}_{SIMULATION}_E={best_energy:.4f}_best.mp4"
                ani.save(fname, writer=writer)
                print(f"✅ Saved MP4: {fname}")
            except Exception as e:
                print(f"MP4 failed: {e}. Trying GIF fallback...")
                writer = PillowWriter(fps=20)
                fname = f"graphs/quark_{MODE}_{SIMULATION}_E={best_energy:.4f}_best.gif"
                ani.save(fname, writer=writer)
                print(f"✅ Saved fallback GIF: {fname}")

    print("\n✅ All done!")
