# fermion_u1_sim.py - Unified Fermion U(1) Graph-Action Simulator
# Models fermions with fixed U(1) charges (photons as implicit mediators via Coulomb)
# Graph: complete graph on N_FERMIONS nodes with positions + fixed charges
# Energy: linear confinement (optional binding) + short-range repulsion + U(1) Coulomb
# No linear/color modes — straight to U(1) fermion model (positions only, charges fixed)
# Runs analogously to quark_sim.py: multi-start gradient descent → best stable configuration

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import os

if not os.path.exists("graphs"):
    os.mkdir("graphs")

# ========================== LOAD CONFIG ==========================
def load_config():
    parser = argparse.ArgumentParser(description="Unified Fermion U(1) Graph-Action Simulator")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--seed", type=int, help="Simulation seed")
    parser.add_argument("--n_fermions", type=int, help="Number of fermions (overrides config)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        print(f"Warning: {config_path} not found. Using defaults.")
        cfg = {}

    # CLI overrides
    if args.n_fermions is not None:
        cfg["N_FERMIONS"] = args.n_fermions
    if args.seed is not None:
        cfg["simulation_seed"] = args.seed

    # Defaults (analogous to quark model + U(1) specifics)
    defaults = {
        "simulation_seed": 42,
        "BOOK_MODE" : True,
        "N_FERMIONS": 4,
        "FERMION_CHARGES": [1.0, -1.0, 1.0, -1.0],  # neutral example (e.g. two +1, two -1)
        "DIM": 3,
        "CONFINEMENT_STRENGTH": 4.0,   # optional binding (set to 0.0 for pure EM)
        "REPULSION_STRENGTH": 2.0,
        "REPULSION_EPS": 0.05,
        "COULOMB_STRENGTH": 8.0,       # U(1) photon-mediated Coulomb strength
        "LEARNING_RATE": 0.07,
        "MAX_ITER": 2000,
        "N_RANDOM_STARTS": 10,
        "DEFAULT_STATIC_ONLY": False,
        "ANIMATE_LIVE": False,
        "SAVE_MP4": True
    }
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v

    # Ensure charges list matches N_FERMIONS
    if len(cfg["FERMION_CHARGES"]) != cfg["N_FERMIONS"]:
        print(f"Warning: FERMION_CHARGES length mismatch. Using first {cfg['N_FERMIONS']} or padding with 0.")
        charges = cfg["FERMION_CHARGES"][:cfg["N_FERMIONS"]]
        charges += [0.0] * (cfg["N_FERMIONS"] - len(charges))
        cfg["FERMION_CHARGES"] = charges

    return cfg

cfg = load_config()

np.random.seed(cfg["simulation_seed"])
torch.manual_seed(cfg["simulation_seed"])

N_FERMIONS = cfg["N_FERMIONS"]
DIM = cfg["DIM"]
CONFINEMENT_STRENGTH = cfg["CONFINEMENT_STRENGTH"]
REPULSION_STRENGTH = cfg["REPULSION_STRENGTH"]
REPULSION_EPS = cfg["REPULSION_EPS"]
COULOMB_STRENGTH = cfg["COULOMB_STRENGTH"]
LEARNING_RATE = cfg["LEARNING_RATE"]
MAX_ITER = cfg["MAX_ITER"]
N_RANDOM_STARTS = cfg["N_RANDOM_STARTS"]
DEFAULT_STATIC_ONLY = cfg["DEFAULT_STATIC_ONLY"]
ANIMATE_LIVE = cfg["ANIMATE_LIVE"]
SAVE_MP4 = cfg["SAVE_MP4"]

print(f"Running → Fermion U(1) model | Fermions: {N_FERMIONS} | Runs: {N_RANDOM_STARTS}")
print(f"   Charges: {cfg['FERMION_CHARGES']}")
print(f"   Coulomb strength: {COULOMB_STRENGTH} | Confinement: {CONFINEMENT_STRENGTH}")

# ========================== GRAPH SETUP ==========================
def create_fermion_graph():
    G = nx.complete_graph(N_FERMIONS)
    charges_list = cfg["FERMION_CHARGES"]
    # Simple labels (user can customize via config if desired)
    labels = [f"f{i}({c:+.1f})" for i, c in enumerate(charges_list)]
    for i in range(N_FERMIONS):
        G.nodes[i]['label'] = labels[i]
        G.nodes[i]['charge'] = charges_list[i]
    charges_tensor = torch.tensor(charges_list, dtype=torch.float32)
    return G, charges_tensor

# ========================== ENERGY ==========================
def total_energy(positions: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
    """U(1) Coulomb (photon-mediated) + confinement + short-range repulsion"""
    E = torch.zeros(1, device=positions.device)

    for i in range(N_FERMIONS):
        for j in range(i + 1, N_FERMIONS):
            r = torch.norm(positions[i] - positions[j])
            # U(1) Coulomb term (photons as implicit force carriers)
            E += COULOMB_STRENGTH * charges[i] * charges[j] / (r + REPULSION_EPS)
            # Confinement (kept for binding analogy; set to 0 in config for pure EM)
            E += CONFINEMENT_STRENGTH * r
            # Short-range repulsion (prevents collapse for opposite charges)
            E += REPULSION_STRENGTH / (r + REPULSION_EPS)

    return E

# ========================== OPTIMIZATION ==========================
def minimize_run(initial_pos_np: np.ndarray, charges: torch.Tensor):
    positions = torch.tensor(initial_pos_np, dtype=torch.float32, requires_grad=True)
    params = [positions]

    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    
    history_pos = []
    energies = []

    for it in tqdm(range(MAX_ITER), desc=f"Optimizing U(1) Fermions"):
        optimizer.zero_grad()
        loss = total_energy(positions, charges)
        loss.backward()
        optimizer.step()

        history_pos.append(positions.detach().clone().numpy())
        energies.append(loss.item())

        if it > 200 and len(energies) > 1 and abs(energies[-1] - energies[-2]) < 1e-6:
            break

    return np.array(history_pos), np.array(energies)

# ========================== PLOTTING ==========================
def plot_3d(G, positions, charges_np, ax, title, energy=None):
    ax.cla()
    
    # Draw nodes colored by charge sign (U(1) charge visualization)
    for i in range(N_FERMIONS):
        charge = charges_np[i]
        if charge > 0:
            col = 'red'      # positive fermions
        elif charge < 0:
            col = 'blue'     # negative fermions
        else:
            col = 'green'    # neutral
        ax.scatter(*positions[i], color=col, s=380, edgecolor='black', linewidth=2.5, zorder=1)

        # Label (zorder=10 ensures visibility, as noted in quark model)
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

    # Draw edges (complete graph connections)
    for i in range(N_FERMIONS):
        for j in range(i + 1, N_FERMIONS):
            p1, p2 = positions[i], positions[j]
            ax.plot(*zip(p1, p2), color='darkgreen', linewidth=3, alpha=0.85, zorder=2)

    # Title with energy (no singlet term needed for U(1))
    info = f" | Net Q = {np.sum(charges_np):.2f}"
    ax.set_title(f"{title}\nE ≈ {energy:.3f}{info}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 13)
    ax.set_ylim(-2, 13)
    ax.set_zlim(-2, 13)

# ========================== MAIN ==========================
if __name__ == "__main__":
    G, charges = create_fermion_graph()
    charges = charges  # fixed U(1) charges (torch tensor)
    
    histories_pos = []
    final_energies = []
    initial_pos_list = []

    print(f"Starting {N_RANDOM_STARTS} runs — Fermion U(1) model")

    for run in range(N_RANDOM_STARTS):
        # Random initial positions (no special clustering needed for general fermion systems)
        init_pos = np.random.rand(N_FERMIONS, DIM) * 6 + 1.0
        initial_pos_list.append(init_pos)
        hist_pos, energies = minimize_run(init_pos, charges)

        histories_pos.append(hist_pos)
        final_energies.append(energies[-1])

        print(f"Run {run+1:2d} → E = {energies[-1]:.4f}")

    best_idx = int(np.argmin(final_energies))
    best_energy = final_energies[best_idx]
    print(f"\n🎯 BEST RUN: #{best_idx+1} with E = {best_energy:.4f}")

    if DEFAULT_STATIC_ONLY:
        print("Generating static figure...")
        cols = N_RANDOM_STARTS + 1
        fig = plt.figure(figsize=(6*cols, 9))
        fig.suptitle(f"Fermion U(1) Simulation — {N_FERMIONS} particles", fontsize=16)

        for r in range(N_RANDOM_STARTS):
            ax = fig.add_subplot(2, cols, r+2, projection='3d')
            plot_3d(G, histories_pos[r][-1], np.array(cfg["FERMION_CHARGES"]), 
                    ax, f"Run {r+1}", final_energies[r])

        # Initial configuration
        ax_init = fig.add_subplot(2, cols, 1, projection='3d')
        plot_3d(G, initial_pos_list[0], np.array(cfg["FERMION_CHARGES"]), 
                ax_init, "Initial", 
                total_energy(torch.tensor(initial_pos_list[0]), charges).item())

        plt.tight_layout()
        plt.savefig(f"graphs/fermion_u1_N{N_FERMIONS}_final.png", dpi=600, bbox_inches='tight')
        print(f"✅ Saved: graphs/fermion_u1_N{N_FERMIONS}_final.png")
        # plt.show()  # uncomment if you want to display

    # Save publication-ready static figure of best configuration
    if cfg.get("BOOK_MODE", False) or True:   # always for now
        fig_best = plt.figure(figsize=(8, 8))
        ax_best = fig_best.add_subplot(111, projection='3d')
        plot_3d(G, histories_pos[best_idx][-1], np.array(cfg["FERMION_CHARGES"]), 
                ax_best, f"Best Fermion U(1) Config (E={best_energy:.4f})", best_energy)
        plt.savefig(f"graphs/fermion_u1_N{N_FERMIONS}_stable.png", dpi=600, bbox_inches='tight')
        print(f"✅ Saved best config: graphs/fermion_u1_N{N_FERMIONS}_stable.png")
        plt.close(fig_best)

    # MP4 of BEST run
    if ANIMATE_LIVE or SAVE_MP4:
        print(f"🎥 Animating BEST run (#{best_idx+1}, E={best_energy:.4f})...")
        hist_pos = histories_pos[best_idx]
        energies = np.array([total_energy(torch.tensor(p), charges).item() for p in hist_pos])

        fig_anim = plt.figure(figsize=(12, 8))
        ax3d = fig_anim.add_subplot(121, projection='3d')
        ax_e = fig_anim.add_subplot(122)
        energy_line, = ax_e.plot([], [], 'b-', linewidth=2)
        ax_e.set_xlim(0, len(energies))
        ax_e.set_ylim(0, max(energies)*1.05)
        ax_e.set_xlabel("Iteration")
        ax_e.set_ylabel("Graph Action E (U(1))")
        ax_e.grid(True)

        def animate(frame):
            pos = hist_pos[frame]
            plot_3d(G, pos, np.array(cfg["FERMION_CHARGES"]), ax3d, f"Frame {frame}", energies[frame])
            energy_line.set_data(range(frame+1), energies[:frame+1])
            return ax3d,

        ani = FuncAnimation(fig_anim, animate, frames=len(hist_pos), interval=40, blit=False)

        if ANIMATE_LIVE:
            plt.show()

        if SAVE_MP4:
            try:
                writer = FFMpegWriter(fps=20, metadata=dict(artist='GraphPhysics'), bitrate=2000)
                fname = f"graphs/fermion_u1_N{N_FERMIONS}_E={best_energy:.4f}_best.mp4"
                ani.save(fname, writer=writer)
                print(f"✅ Saved MP4: {fname}")
            except Exception as e:
                print(f"MP4 failed: {e}. Trying GIF fallback...")
                writer = PillowWriter(fps=20)
                fname = f"graphs/fermion_u1_N{N_FERMIONS}_E={best_energy:.4f}_best.gif"
                ani.save(fname, writer=writer)
                print(f"✅ Saved fallback GIF: {fname}")

    print("\n✅ All done! Stable U(1) fermion configuration found via graph-action minimization.")
