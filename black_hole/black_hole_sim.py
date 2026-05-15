# black_hole_sim.py - Entanglement Graph Black Hole Simulation
# Demonstrates horizon formation, evaporation, Page curve, and Planck-scale deviations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# ========================= CONFIG =========================
N_SIDE = 14
MASS_INCREASE_RATE = 0.12
EVAPORATION_START = 90
EVAPORATION_RATE = 0.008
MAX_ITER = 280

# Planck-scale parameters
PLANCK_SCALE_FLUCTUATION = 0.018
MEASURE_DEVIATIONS = True

np.random.seed(42)

# ========================= GRAPH SETUP =========================
def create_2d_lattice():
    G = nx.grid_2d_graph(N_SIDE, N_SIDE)
    # Add diagonal edges for better connectivity (8-connectivity)
    for x in range(N_SIDE):
        for y in range(N_SIDE):
            for dx, dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < N_SIDE and 0 <= ny_ < N_SIDE:
                    G.add_edge((x, y), (nx_, ny_), weight=1.0)
    
    pos = {node: np.array(node, dtype=float) for node in G.nodes()}
    
    # Initial mass with central seed
    mass = np.ones((N_SIDE, N_SIDE)) * 0.08
    cx, cy = N_SIDE//2, N_SIDE//2
    for i in range(-4, 5):
        for j in range(-4, 5):
            r = np.hypot(i, j)
            mass[cx + i, cy + j] = max(mass[cx + i, cy + j], 0.65 * np.exp(-r / 2.8))
    
    return G, pos, mass

G, pos, mass = create_2d_lattice()
center = np.array([N_SIDE / 2, N_SIDE / 2], dtype=float)

# ========================= HELPERS =========================
def get_neighbors(node):
    return list(G.neighbors(node))

def local_curvature(node, mass):
    x, y = node
    neighbors = get_neighbors(node)
    if not neighbors:
        return 0.0
    m_local = mass[x, y]
    m_neighbors = np.array([mass[nx, ny] for nx, ny in neighbors])
    avg_m = m_neighbors.mean()
    curvature = 1.25 * (m_local - avg_m) + 0.65 * (m_local - 0.12)
    return np.clip(curvature, -0.5, 2.8)

def local_curvature_map(mass):
    """Simple curvature proxy for fluctuation strength"""
    curv = np.zeros_like(mass)
    for x in range(N_SIDE):
        for y in range(N_SIDE):
            node = (x, y)
            neighbors = get_neighbors(node)
            if neighbors:
                m_local = mass[x, y]
                m_nb = np.mean([mass[nx, ny] for nx, ny in neighbors])
                curv[x, y] = abs(m_local - m_nb)
    return curv

# ========================= SIMULATION =========================
history_mass = []
history_horizon = []
history_bidirect = []
history_S_rad = []
history_S_bh = []
history_deviation = []

print("Running black hole simulation with Page curve + Planck-scale fluctuations...")

for t in tqdm(range(MAX_ITER)):
    cx, cy = int(center[0]), int(center[1])
    
    # === 1. Mass dynamics ===
    if t < EVAPORATION_START:
        xx, yy = np.indices(mass.shape)
        r = np.hypot(xx - cx, yy - cy)
        growth = MASS_INCREASE_RATE * (1 - t / 80) * np.exp(-r / 4.0)
        mass += growth
    else:
        mass *= (1 - EVAPORATION_RATE)
        if t % 4 == 0:
            mass = gaussian_filter(mass, sigma=1.1)
            xx, yy = np.indices(mass.shape)
            r = np.hypot(xx - cx, yy - cy)
            noise = np.random.normal(0, 0.0045, mass.shape)
            noise *= (r / (r.max() + 1e-8) + 0.2)
            mass += noise
    
    # === Planck-scale quantum fluctuations ===
    if PLANCK_SCALE_FLUCTUATION > 0:
        noise = np.random.normal(0, PLANCK_SCALE_FLUCTUATION, mass.shape)
        noise *= (1 + 0.8 * local_curvature_map(mass))
        mass += noise
    
    mass = np.clip(mass, 0.04, 9.0)
    
    # === 2. Trapped surface detection ===
    radii = []
    for rad in range(1, N_SIDE//2 + 1):
        shell = [(x, y) for x in range(N_SIDE) for y in range(N_SIDE)
                 if int(np.hypot(x - cx, y - cy)) == rad]
        if len(shell) < 10:
            continue
        curv = [local_curvature(node, mass) for node in shell]
        trapped_frac = np.mean(np.array(curv) > 0.23)
        if trapped_frac > 0.52:
            radii.append(rad)
    
    horizon_r = np.mean(radii) if radii else 1.8
    history_horizon.append(horizon_r)
    
    # === 3. Causal bidirectionality ===
    bidirectional = 0
    total = 0
    for u in G.nodes():
        ux, uy = u
        dist_u = np.hypot(ux - cx, uy - cy)
        for v in get_neighbors(u):
            total += 1
            vx, vy = v
            dist_v = np.hypot(vx - cx, vy - cy)
            if max(dist_u, dist_v) < horizon_r - 0.7:
                continue
            bidirectional += 1
    
    bidirect_frac = bidirectional / total if total > 0 else 0
    history_bidirect.append(bidirect_frac)
    
    # === 4. Page curve proxy ===
    xx, yy = np.indices(mass.shape)
    r = np.hypot(xx - cx, yy - cy)
    inside = r < horizon_r
    M_bh = mass[inside].sum()
    M_rad = mass[~inside].sum()
    
    S_bh = (np.pi * horizon_r**2) * np.log(1 + M_bh)
    S_rad = M_rad * np.log(1 + M_rad + 1e-6)
    history_S_rad.append(S_rad)
    history_S_bh.append(S_bh)
    
    # === 5. Planck deviation ===
    if MEASURE_DEVIATIONS:
        classical_hr = 0.6 * np.sqrt(M_bh) if M_bh > 0 else horizon_r
        deviation = abs(horizon_r - classical_hr) / (classical_hr + 1e-6)
        history_deviation.append(deviation)
    
    history_mass.append(mass.copy())

# ========================= ANIMATION =========================
fig, axs = plt.subplots(2, 2, figsize=(15, 11))
axs = axs.ravel()

# Mass map
im = axs[0].imshow(history_mass[0].T, origin='lower', cmap='magma', 
                   vmin=0, vmax=5, animated=True)
axs[0].set_title("Mass Distribution + Curvature (t=0)")
horizon_circle = plt.Circle((center[0], center[1]), 3.0, color='cyan', 
                           fill=False, linewidth=3.8, alpha=0.9, animated=True)
axs[0].add_patch(horizon_circle)
plt.colorbar(im, ax=axs[0], fraction=0.046)

# Graph
axs[1].set_title("Entanglement Graph (Central Region)")
sub_nodes = [(x,y) for x in range(max(0,cx-6), min(N_SIDE,cx+7)) 
                    for y in range(max(0,cy-6), min(N_SIDE,cy+7))]
node_collection = nx.draw_networkx_nodes(G.subgraph(sub_nodes), pos=pos, ax=axs[1],
    node_color=[history_mass[0][n] for n in sub_nodes],
    cmap='magma', node_size=110, edgecolors='black', linewidths=0.8)
nx.draw_networkx_edges(G.subgraph(sub_nodes), pos=pos, ax=axs[1], 
                      edge_color='gray', alpha=0.55, width=1.1)

graph_horizon = plt.Circle((center[0], center[1]), 3.0, color='cyan', 
                          fill=False, linewidth=2.2, alpha=0.7)
axs[1].add_patch(graph_horizon)

# Time series
line_h, = axs[2].plot([], [], 'c-', lw=2.8)
axs[2].set_title("Event Horizon Radius")
axs[2].set_ylabel("Radius (lattice units)")
axs[2].grid(True, alpha=0.3)
axs[2].set_ylim(0, N_SIDE//2 + 1)   # Much better range
axs[2].set_xlim(0, MAX_ITER)

line_b, = axs[3].plot([], [], 'lime', lw=2.8)
axs[3].set_title("Causal Bidirectionality Fraction")
axs[3].set_ylabel("Fraction")
axs[3].set_ylim(0, 1)
axs[3].grid(True, alpha=0.3)

fig.suptitle("Entanglement Graph Black Hole Simulation — Improved", fontsize=16, y=0.98)

def animate(frame):
    m = history_mass[frame]
    hr = history_horizon[frame]
    bf = history_bidirect[frame]
    
    im.set_array(m.T)
    axs[0].set_title(f"Mass Distribution + Curvature (t={frame})")
    horizon_circle.set_radius(hr)
    
    # Update graph colors
    colors = [m[node] for node in sub_nodes]
    node_collection.set_array(np.array(colors))
    node_collection.set_clim(0, 5)
    
    graph_horizon.set_radius(hr)
    
    # Update lines
    line_h.set_data(range(frame+1), history_horizon[:frame+1])
    line_b.set_data(range(frame+1), history_bidirect[:frame+1])
    
    axs[2].set_xlim(0, frame + 20)
    axs[3].set_xlim(0, frame + 20)
    
    return [im, horizon_circle, node_collection, line_h, line_b]

ani = FuncAnimation(fig, animate, frames=MAX_ITER, interval=65, blit=False, repeat=True)

# Save
os.makedirs("graphs", exist_ok=True)
print("\nSaving animation...")
output_filepath = f"graphs/black_hole,N={N_SIDE}.mp4"
ani.save(output_filepath, writer='ffmpeg', fps=15, dpi=170)
print(f"✅ Saved: {output_filepath}")

# Final frame
animate(MAX_ITER-1)
plt.savefig("graphs/black_hole_final.png", dpi=280, bbox_inches='tight')
plt.show()

# ========================= PAGE CURVE =========================
print("\nGenerating Page curve...")
fig_page, ax_page = plt.subplots(figsize=(10, 6))
t_arr = np.arange(MAX_ITER)
ax_page.plot(t_arr, history_S_rad, 'r-', lw=2.5, label=r'Radiation Entropy $S_{\rm rad}(t)$')
ax_page.plot(t_arr, history_S_bh, 'b--', lw=2, label=r'Black Hole Entropy $S_{\rm BH}(t)$')
ax_page.plot(t_arr, np.array(history_S_rad) + np.array(history_S_bh), 'k:', lw=1.5, 
             alpha=0.7, label='Total Entropy (approx. conserved)')
ax_page.axvline(EVAPORATION_START, color='gray', linestyle='--', label='Evaporation onset')
ax_page.set_xlabel('Time $t$')
ax_page.set_ylabel('Entropy (arbitrary units)')
ax_page.set_title('Emergent Page Curve from Entanglement Graph Simulation')
ax_page.legend()
ax_page.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/page_curve_proxy.png", dpi=300, bbox_inches='tight')
print("✅ Saved: graphs/page_curve_proxy.png")

# ========================= PLANCK-SCALE DEVIATIONS =========================
if MEASURE_DEVIATIONS:
    print("Generating Planck-scale deviation plot...")
    fig_dev, ax_dev = plt.subplots(figsize=(10, 6))
    ax_dev.plot(t_arr, history_deviation, 'm-', lw=2.5, label='Horizon Radius Deviation')
    ax_dev.set_xlabel('Time $t$')
    ax_dev.set_ylabel('Relative Deviation from Classical')
    ax_dev.set_title('Planck-Scale Deviations in Entanglement Graph Black Hole')
    ax_dev.legend()
    ax_dev.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("graphs/planck_deviations.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: graphs/planck_deviations.png")

# Final static frame
print("Saving final frame...")
animate(MAX_ITER-1)
plt.savefig("graphs/black_hole_final.png", dpi=280, bbox_inches='tight')

print("\n✅ All simulations complete!")
