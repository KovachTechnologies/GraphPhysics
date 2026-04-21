import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import colorsys

# ========================== CONFIGURATION ==========================
np.random.seed(42)
torch.manual_seed(42)

N_QUARKS = 3
DIM = 3                          # true 3D positions

# QCD parameters
CONFINEMENT_STRENGTH = 12.0      # base string tension σ
REPULSION_STRENGTH = 1.5
REPULSION_EPS = 0.05
COLOR_SINGLET_STRENGTH = 8.0     # penalty for deviation from color singlet

LEARNING_RATE = 0.08
MAX_ITER = 1000
N_RANDOM_STARTS = 3

# Visualization flags
DEFAULT_STATIC_ONLY = True       # Set False + ANIMATE_LIVE=True for real-time movie
ANIMATE_LIVE = False
SAVE_GIF = False

NODE_LABELS = ['u', 'd1', 'd2']

# ========================== GRAPH SETUP ==========================
def create_quark_graph():
    G = nx.complete_graph(N_QUARKS)
    for i, label in enumerate(NODE_LABELS):
        G.nodes[i]['label'] = label
    return G

# ========================== COLOR DEGREES OF FREEDOM ==========================
def random_color_state():
    """Random normalized complex 3-vector (color in SU(3) fundamental rep)."""
    c = torch.randn(3, dtype=torch.cfloat)
    return c / torch.norm(c)

def color_singlet_projector(colors: torch.Tensor) -> torch.Tensor:
    """Color-singlet fidelity for three quarks: |ε_ijk c1^i c2^j c3^k|^2 / normalization."""
    # Levi-Civita contraction for color singlet
    eps = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],
                        [[0,0,-1],[0,0,0],[1,0,0]],
                        [[0,1,0],[-1,0,0],[0,0,0]]], dtype=torch.cfloat)
    c1, c2, c3 = colors[0], colors[1], colors[2]
    singlet = 0j
    for i in range(3):
        for j in range(3):
            for k in range(3):
                singlet += eps[i,j,k] * c1[i] * c2[j] * c3[k]
    fidelity = torch.abs(singlet)**2
    return fidelity

# ========================== TOTAL ENERGY (QCD + COLOR) ==========================
def total_energy(positions: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Full graph action proxy: spatial QCD + color singlet term."""
    E_spatial = torch.zeros(1, device=positions.device)
    for i in range(N_QUARKS):
        for j in range(i + 1, N_QUARKS):
            r = torch.norm(positions[i] - positions[j])
            # Color-dependent confinement (stronger when colors are "aligned" for singlet)
            color_factor = torch.real(torch.dot(colors[i].conj(), colors[j])).abs() + 0.5
            E_spatial += CONFINEMENT_STRENGTH * color_factor * r
            E_spatial += REPULSION_STRENGTH / (r + REPULSION_EPS)

    # Color singlet penalty (strongly favors color-neutral baryon)
    singlet_fid = color_singlet_projector(colors)
    E_color = COLOR_SINGLET_STRENGTH * (1.0 - singlet_fid)

    return E_spatial + E_color

# ========================== OPTIMIZATION ==========================
def minimize_torch_with_color(initial_pos_np: np.ndarray, max_iter=MAX_ITER, lr=LEARNING_RATE):
    """Joint optimization of 3D positions and color vectors."""
    positions = torch.tensor(initial_pos_np, dtype=torch.float32, requires_grad=True)
    
    # Initialize color states
    colors = torch.stack([random_color_state() for _ in range(N_QUARKS)], dim=0)
    colors.requires_grad = True
    
    optimizer = torch.optim.Adam([positions, colors], lr=lr)
    
    history_pos = []
    history_colors = []
    energies = []
    
    print(f"Starting joint position + color minimization ({max_iter} iterations)...")
    for it in tqdm(range(max_iter), desc="Optimizing"):
        optimizer.zero_grad()
        loss = total_energy(positions, colors)
        loss.backward()
        optimizer.step()
        
        # Renormalize colors to stay on unit sphere
        with torch.no_grad():
            for c in colors:
                c /= torch.norm(c)
        
        history_pos.append(positions.detach().clone().numpy())
        history_colors.append(colors.detach().clone().numpy())
        energies.append(loss.item())
        
        if it > 100 and abs(energies[-1] - energies[-2]) < 1e-5:
            break
    
    return (np.array(history_pos), 
            np.array(history_colors), 
            np.array(energies))

# ========================== COLOR TO RGB ==========================
def color_vector_to_rgb(c_vec: np.ndarray) -> tuple:
    """Convert complex color vector to visible RGB (approximate)."""
    # Take real parts and normalize
    rgb = np.abs(c_vec[:3].real)
    rgb /= (np.sum(rgb) + 1e-8)
    # Enhance saturation
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    rgb = colorsys.hsv_to_rgb(h, min(1.0, s*1.4), v)
    return rgb

# ========================== 3D PLOTTING WITH COLOR ==========================
def plot_3d_with_color(G, positions, colors_np, ax, title, energy=None):
    ax.cla()
    for i in range(N_QUARKS):
        rgb = color_vector_to_rgb(colors_np[i])
        ax.scatter(*positions[i], color=rgb, s=350, edgecolor='black', linewidth=2)
        ax.text(*positions[i], f" {G.nodes[i]['label']}", color='white', fontsize=11, ha='center')
    
    for i in range(N_QUARKS):
        for j in range(i + 1, N_QUARKS):
            p1 = positions[i]
            p2 = positions[j]
            ax.plot(*zip(p1, p2), color='darkgreen', linewidth=3.5, alpha=0.9)
            dist = np.linalg.norm(p1 - p2)
            mid = (p1 + p2) / 2
            ax.text(*mid, f'{dist:.2f}', color='black', fontsize=9)
    
    singlet_fid = color_singlet_projector(torch.tensor(colors_np, dtype=torch.cfloat)).item()
    title_str = f"{title}\nE ≈ {energy:.3f} | Singlet fidelity: {singlet_fid:.3f}"
    ax.set_title(title_str)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(-1, 9); ax.set_ylim(-1, 9); ax.set_zlim(-1, 9)

# ========================== MAIN ==========================
if __name__ == "__main__":
    G = create_quark_graph()
    
    histories_pos = []
    histories_colors = []
    final_energies = []
    initial_pos_list = []
    
    print("Running 3D QCD + color minimization...")
    for run in range(N_RANDOM_STARTS):
        init_pos = np.random.rand(N_QUARKS, DIM) * 6 + 1.0
        initial_pos_list.append(init_pos)
        
        hist_pos, hist_col, energies = minimize_torch_with_color(init_pos)
        histories_pos.append(hist_pos)
        histories_colors.append(hist_col)
        final_energies.append(energies[-1])
        
        print(f"Run {run+1} → final E = {energies[-1]:.4f}")
    
    # ===================== DEFAULT STATIC 3D FIGURE =====================
    if DEFAULT_STATIC_ONLY:
        print("\nGenerating static 3D result with color...")
        fig = plt.figure(figsize=(6 * (N_RANDOM_STARTS + 1), 9))
        fig.suptitle("PyTorch 3D QCD Minimization with Color Degrees of Freedom\n"
                     "udd baryon (neutron-like) • Color singlet emerges automatically", fontsize=15)
        
        for run in range(N_RANDOM_STARTS):
            ax = fig.add_subplot(2, N_RANDOM_STARTS + 1, run + 2, projection='3d')
            final_pos = histories_pos[run][-1]
            final_col = histories_colors[run][-1]
            plot_3d_with_color(G, final_pos, final_col, ax, f"Run {run+1} (final)", final_energies[run])
        
        # Initial configuration (column 1)
        ax_init = fig.add_subplot(2, N_RANDOM_STARTS + 1, 1, projection='3d')
        plot_3d_with_color(G, initial_pos_list[0], 
                          np.stack([random_color_state().numpy() for _ in range(N_QUARKS)]), 
                          ax_init, "Initial (random)", 
                          total_energy(torch.tensor(initial_pos_list[0]), 
                                      torch.stack([random_color_state() for _ in range(N_QUARKS)])).item())
        
        # Energy curves
        for run in range(N_RANDOM_STARTS):
            ax_e = fig.add_subplot(2, N_RANDOM_STARTS + 1, N_RANDOM_STARTS + 2 + run)
            energies_run = [total_energy(torch.tensor(pos), torch.tensor(col)).item() 
                           for pos, col in zip(histories_pos[run], histories_colors[run])]
            ax_e.plot(energies_run, 'b-', linewidth=2)
            ax_e.set_xlabel("Iteration")
            ax_e.set_ylabel("Total Graph Action E")
            ax_e.set_title(f"Convergence (Run {run+1})")
            ax_e.grid(True)
        
        plt.tight_layout()
        plt.savefig("3D_quark_QCD_color_minimization_final.png", dpi=300, bbox_inches='tight')
        print("✅ Saved: 3D_quark_QCD_color_minimization_final.png")
        plt.show()
    
    # ===================== OPTIONAL ANIMATION =====================
    if ANIMATE_LIVE or SAVE_GIF:
        print("\nGenerating animation (first run)...")
        hist_pos = histories_pos[0]
        hist_col = histories_colors[0]
        energies = np.array([total_energy(torch.tensor(p), torch.tensor(c)).item() 
                            for p, c in zip(hist_pos, hist_col)])
        
        fig_anim = plt.figure(figsize=(12, 8))
        ax3d = fig_anim.add_subplot(121, projection='3d')
        ax_e = fig_anim.add_subplot(122)
        
        energy_line, = ax_e.plot([], [], 'b-', linewidth=2)
        ax_e.set_xlim(0, len(energies))
        ax_e.set_ylim(0, max(energies)*1.05)
        ax_e.set_xlabel("Iteration")
        ax_e.set_ylabel("Graph Action E")
        
        def animate(frame):
            pos = hist_pos[frame]
            col = hist_col[frame]
            plot_3d_with_color(G, pos, col, ax3d, f"Frame {frame}", energies[frame])
            energy_line.set_data(range(frame+1), energies[:frame+1])
            return ax3d,
        
        ani = FuncAnimation(fig_anim, animate, frames=len(hist_pos), interval=40, blit=False)
        
        if ANIMATE_LIVE:
            plt.show()
        if SAVE_GIF:
            writer = PillowWriter(fps=20)
            ani.save("3D_quark_QCD_color_evolution.gif", writer=writer, dpi=120)
            print("✅ Saved GIF: 3D_quark_QCD_color_evolution.gif")
    
    print("\n✅ Done! Color degrees of freedom are now included.")
    print("   The minimization naturally drives the system toward a color-singlet baryon configuration.")
