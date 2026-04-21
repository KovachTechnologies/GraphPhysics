import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from tqdm import tqdm

# ========================== CONFIGURATION ==========================
np.random.seed(42)
torch.manual_seed(42)

N_QUARKS = 3
CONFINEMENT_STRENGTH = 10.0      # σ (linear confinement from gluons)
REPULSION_STRENGTH = 1.0         # asymptotic freedom / Coulomb-like repulsion
REPULSION_EPS = 0.05
LEARNING_RATE = 0.08
MAX_ITER = 800
N_RANDOM_STARTS = 3              # number of independent runs for static demo
DIM = 3                          # true 3D

# Flags for visualization
DEFAULT_STATIC_ONLY = True       # ← default: only finished product (recommended first run)
ANIMATE_LIVE = False             # set True for real-time 3D animation during optimization
SAVE_GIF = False                 # set True to also save animated GIF of one run

NODE_LABELS = ['u', 'd1', 'd2']  # neutron-like udd configuration

# ========================== GRAPH SETUP ==========================
def create_quark_graph():
    G = nx.complete_graph(N_QUARKS)
    for i, label in enumerate(NODE_LABELS):
        G.nodes[i]['label'] = label
    return G

# ========================== TORCH ENERGY (QCD-ONLY) ==========================
def quark_energy_torch(positions: torch.Tensor) -> torch.Tensor:
    """Pure QCD effective potential (no gravity). Linear confinement + short-range repulsion."""
    E = torch.zeros(1, device=positions.device)
    for i in range(N_QUARKS):
        for j in range(i + 1, N_QUARKS):
            r = torch.norm(positions[i] - positions[j])
            E += CONFINEMENT_STRENGTH * r                          # gluon string tension
            E += REPULSION_STRENGTH / (r + REPULSION_EPS)         # asymptotic freedom
    return E

# ========================== OPTIMIZATION ==========================
def minimize_torch(initial_pos_np: np.ndarray, max_iter=MAX_ITER, lr=LEARNING_RATE):
    """PyTorch gradient descent on 3D quark positions."""
    positions = torch.tensor(initial_pos_np, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([positions], lr=lr)
    
    history = []
    energies = []
    
    print(f"Starting minimization (3D, {max_iter} iterations)...")
    for it in tqdm(range(max_iter), desc="Optimizing"):
        optimizer.zero_grad()
        loss = quark_energy_torch(positions)
        loss.backward()
        optimizer.step()
        
        history.append(positions.detach().clone().numpy())
        energies.append(loss.item())
        
        # Early stopping
        if it > 50 and abs(energies[-1] - energies[-2]) < 1e-6:
            break
    
    return np.array(history), np.array(energies)

# ========================== 3D PLOTTING ==========================
def plot_3d(G, positions, ax, title, energy=None):
    ax.cla()
    pos_dict = {i: positions[i] for i in range(N_QUARKS)}
    
    # Nodes
    colors = ['blue', 'red', 'red']
    for i in range(N_QUARKS):
        ax.scatter(*pos_dict[i], color=colors[i], s=300, edgecolor='black', linewidth=1.5)
        ax.text(*pos_dict[i], f" {G.nodes[i]['label']}", color='white', fontsize=12, ha='center')
    
    # Edges + distance labels
    for i in range(N_QUARKS):
        for j in range(i + 1, N_QUARKS):
            p1 = np.array(pos_dict[i])
            p2 = np.array(pos_dict[j])
            ax.plot(*zip(p1, p2), color='darkgreen', linewidth=4, alpha=0.85)
            dist = np.linalg.norm(p1 - p2)
            mid = (p1 + p2) / 2
            ax.text(*mid, f'{dist:.2f}', color='black', fontsize=9, ha='center')
    
    ax.set_title(f"{title}\nE ≈ {energy:.3f}" if energy is not None else title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_zlim(-1, 8)

# ========================== ANIMATION UPDATE ==========================
def animate_3d(frame, G, history, ax, energy_line, energies):
    pos = history[frame]
    plot_3d(G, pos, ax, f"3-Quark Minimization (frame {frame})", energies[frame])
    energy_line.set_data(range(frame + 1), energies[:frame + 1])
    return ax,

# ========================== MAIN ==========================
if __name__ == "__main__":
    G = create_quark_graph()
    
    # Run multiple independent optimizations
    histories = []
    final_energies = []
    initial_pos_list = []
    
    print("Running multiple 3D QCD minimizations...")
    for run in range(N_RANDOM_STARTS):
        # Random initial positions in a box
        init_pos = np.random.rand(N_QUARKS, DIM) * 6 + 1
        initial_pos_list.append(init_pos)
        
        history, energies = minimize_torch(init_pos, MAX_ITER, LEARNING_RATE)
        histories.append(history)
        final_energies.append(energies[-1])
        
        print(f"Run {run+1}/{N_RANDOM_STARTS} → final E = {energies[-1]:.4f}")
    
    # ===================== DEFAULT: STATIC FINISHED PRODUCT =====================
    if DEFAULT_STATIC_ONLY:
        print("\nGenerating static 3D finished-product figure...")
        fig = plt.figure(figsize=(6 * (N_RANDOM_STARTS + 1), 8))
        fig.suptitle("PyTorch 3D QCD Graph-Action Minimization\n"
                     "udd baryon (neutron-like) • Pure linear confinement + repulsion", fontsize=16)
        
        # Row 1: 3D final configurations
        for run in range(N_RANDOM_STARTS):
            ax = fig.add_subplot(2, N_RANDOM_STARTS + 1, run + 2, projection='3d')
            final_pos = histories[run][-1]
            plot_3d(G, final_pos, ax, f"Run {run+1} (final)", final_energies[run])
        
        # Initial reference (first column)
        ax_init = fig.add_subplot(2, N_RANDOM_STARTS + 1, 1, projection='3d')
        plot_3d(G, initial_pos_list[0], ax_init, "Initial (random)", quark_energy_torch(torch.tensor(initial_pos_list[0])).item())
        
        # Row 2: Energy convergence curves
        for run in range(N_RANDOM_STARTS):
            ax_e = fig.add_subplot(2, N_RANDOM_STARTS + 1, N_RANDOM_STARTS + 2 + run)
            
            # Recompute energies from stored positions (more accurate than the optimizer's energies)
            energies_run = [quark_energy_torch(torch.tensor(pos)).item() 
                           for pos in histories[run]]
            
            ax_e.plot(energies_run, 'b-', linewidth=2, label='Energy')
            ax_e.axhline(y=final_energies[run], color='r', linestyle='--', alpha=0.7, 
                        label=f'Final E = {final_energies[run]:.4f}')
            
            ax_e.set_xlabel("Iteration")
            ax_e.set_ylabel("Graph Action E")
            ax_e.set_title(f"Convergence (Run {run+1})")
            ax_e.grid(True)
            ax_e.legend()
        
        plt.tight_layout()
        plt.savefig("3D_quark_QCD_minimization_final.png", dpi=300, bbox_inches='tight')
        print("✅ Saved static result: 3D_quark_QCD_minimization_final.png")
        plt.show()
    
    # ===================== OPTIONAL: LIVE ANIMATION + GIF =====================
    if ANIMATE_LIVE or SAVE_GIF:
        print("\nGenerating animation for one run...")
        history = histories[0]           # first run for demo
        energies = np.array([quark_energy_torch(torch.tensor(pos)).item() for pos in history])
        
        fig_anim = plt.figure(figsize=(10, 8))
        ax3d = fig_anim.add_subplot(121, projection='3d')
        ax_e = fig_anim.add_subplot(122)
        
        # Initial energy line
        energy_line, = ax_e.plot([], [], 'b-', linewidth=2)
        ax_e.set_xlim(0, len(history))
        ax_e.set_ylim(0, max(energies) * 1.1)
        ax_e.set_xlabel("Iteration")
        ax_e.set_ylabel("Graph Action E")
        ax_e.grid(True)
        
        ani = FuncAnimation(fig_anim, animate_3d, frames=len(history), 
                            fargs=(G, history, ax3d, energy_line, energies),
                            interval=40, blit=False)
        
        if ANIMATE_LIVE:
            plt.show()
        
        if SAVE_GIF:
            writer = PillowWriter(fps=25)
            ani.save("3D_quark_QCD_evolution.gif", writer=writer, dpi=150)
            print("✅ Saved GIF: 3D_quark_QCD_evolution.gif")
    
    print("\nDone! The final 3D configuration is the stable color-singlet baryon predicted by the graph-action minimization principle.")
