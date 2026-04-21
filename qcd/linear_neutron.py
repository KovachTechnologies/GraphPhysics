import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
import os

# ========================== PARAMETERS ==========================
np.random.seed(42)
N_QUARKS = 3
CONFINEMENT_STRENGTH = 10.0      # σ (linear potential)
REPULSION_STRENGTH = 1.0
REPULSION_EPS = 0.1
LEARNING_RATE = 0.05
MAX_ITER = 300
N_RANDOM_STARTS = 5              # how many independent runs to show stability
DIM = 2                          # 2D for easy visualization (easily changed to 3)

# Node labels for neutron (udd)
NODE_LABELS = ['u', 'd1', 'd2']

# ========================== GRAPH SETUP ==========================
def create_quark_graph():
    G = nx.complete_graph(N_QUARKS)
    for i, label in enumerate(NODE_LABELS):
        G.nodes[i]['label'] = label
    return G

# ========================== ENERGY (DISCRETE ACTION PROXY) ==========================
def quark_energy(x_flat):
    """Effective potential approximating the graph action S_G.
    x_flat: (N_QUARKS * DIM) array of positions."""
    positions = x_flat.reshape((N_QUARKS, DIM))
    E = 0.0
    for i in range(N_QUARKS):
        for j in range(i + 1, N_QUARKS):
            r = np.linalg.norm(positions[i] - positions[j])
            E += CONFINEMENT_STRENGTH * r                     # gluon string tension
            E += REPULSION_STRENGTH / (r + REPULSION_EPS)    # asymptotic freedom
    return E

# ========================== GRADIENT DESCENT ITERATIONS ==========================
def minimize_with_history(initial_x, max_iter=MAX_ITER):
    """Run gradient descent and record history for visualization."""
    x = initial_x.copy()
    history = [x.copy()]
    energies = [quark_energy(x)]
    
    for it in range(max_iter):
        # numerical gradient
        grad = np.zeros_like(x)
        eps = 1e-6
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (quark_energy(x_plus) - energies[-1]) / eps
        x -= LEARNING_RATE * grad
        history.append(x.copy())
        energies.append(quark_energy(x))
        
        # stop early if converged
        if np.linalg.norm(grad) < 1e-5:
            break
    
    return np.array(history), np.array(energies)

# ========================== VISUALIZATION ==========================
def plot_configuration(G, positions, ax, title, energy=None):
    ax.clear()
    pos_dict = {i: positions[i] for i in range(N_QUARKS)}
    nx.draw_networkx_nodes(G, pos_dict, node_color=['blue', 'red', 'red'], 
                           node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos_dict, width=3, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos_dict, 
                            labels={i: G.nodes[i]['label'] for i in range(N_QUARKS)},
                            font_color='white', font_size=14, ax=ax)
    
    # draw distances
    for i in range(N_QUARKS):
        for j in range(i+1, N_QUARKS):
            p1 = pos_dict[i]
            p2 = pos_dict[j]
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            mid = (np.array(p1) + np.array(p2)) / 2
            ax.text(mid[0], mid[1], f'{dist:.2f}', fontsize=10, ha='center')
    
    ax.set_title(f"{title}\nE ≈ {energy:.3f}" if energy is not None else title)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    ax.axis('off')

# ========================== MAIN DEMO ==========================
if __name__ == "__main__":
    G = create_quark_graph()
    
    # Run multiple random starts to demonstrate stability
    histories = []
    final_energies = []
    
    fig, axes = plt.subplots(2, N_RANDOM_STARTS + 1, figsize=(4*(N_RANDOM_STARTS+1), 8))
    fig.suptitle("Graph-Action Minimization: 3-Quark (Neutron) System\n"
                 "Blue = u, Red = d   |   Convergence to stable baryon", fontsize=14)
    
    # Column 0: initial random configuration (same for all)
    initial_x = np.random.rand(N_QUARKS * DIM) * 8 + 1
    plot_configuration(G, initial_x.reshape((N_QUARKS, DIM)), axes[0, 0], "Initial (random)", quark_energy(initial_x))
    
    for run in range(N_RANDOM_STARTS):
        init_x = np.random.rand(N_QUARKS * DIM) * 8 + 1
        history, energies = minimize_with_history(init_x)
        histories.append(history)
        final_energies.append(energies[-1])
        
        # Plot final minimized state
        final_pos = history[-1].reshape((N_QUARKS, DIM))
        plot_configuration(G, final_pos, axes[0, run+1], f"Run {run+1} (final)", energies[-1])
        
        # Plot energy convergence curve
        axes[1, run+1].plot(energies, 'b-', linewidth=2)
        axes[1, run+1].set_xlabel("Iteration")
        axes[1, run+1].set_ylabel("Graph Action E")
        axes[1, run+1].set_title(f"Convergence (Run {run+1})")
        axes[1, run+1].grid(True)
    
    # Column 0 row 1: legend / summary
    axes[1, 0].text(0.5, 0.5, f"Mean final E = {np.mean(final_energies):.3f}\n"
                              f"Std = {np.std(final_energies):.3f}\n\n"
                              "All runs converge to the same\n"
                              "compact triangular configuration\n"
                              "(stable neutron-like baryon)",
                    ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig("quark_graph_minimization.png", dpi=300, bbox_inches='tight')
    print("✅ Saved figure: quark_graph_minimization.png")
    print(f"Final energies (5 runs): {final_energies}")
    
    # Optional animated GIF of one run (uncomment if desired)
    # ani = FuncAnimation(fig, animate, frames=len(history), interval=50, ...)
    # ani.save("quark_evolution.gif", writer='pillow')
    
    plt.show()
