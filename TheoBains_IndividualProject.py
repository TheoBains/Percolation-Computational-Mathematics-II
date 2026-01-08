import numpy as np
from matplotlib import pyplot as plt
import heapq
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# PART 1: The Optimized Algorithm (Min-Heap)
# ==========================================

def run_invasion_percolation(L, dim=2, max_steps=None):
    """
    Simulates Invasion Percolation on a lattice of size L^dim.
    Returns:
        - cluster_grid: The final lattice (0=empty, >0 = step invaded)
        - stats: Dictionary containing history of Mass (M), Rg, and invaded resistances (r)
    """
    # 1. Initialize Grid and Resistances
    shape = tuple([L] * dim)
    # Assign random resistances r in [0, 1] for every site
    resistances = np.random.rand(*shape)
    
    # Grid to track invasion time: 0 = uninvaded, 1, 2, 3... = step number
    cluster_grid = np.zeros(shape, dtype=int)
    
    # Min-Heap for the boundary: Stores tuples (resistance, x, y, [z])
    boundary = []
    
    # Set to keep track of sites currently in the heap to prevent duplicates
    boundary_set = set()
    
    # 2. Seed the Center
    center = tuple([L // 2] * dim)
    cluster_grid[center] = 1 # Step 1
    
    # Define neighbors (Von Neumann neighborhood)
    if dim == 2:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), 
                     (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    
    # Add initial neighbors to heap
    for delta in neighbors:
        n_coord = tuple(np.array(center) + np.array(delta))
        if all(0 <= c < L for c in n_coord):
            r_val = resistances[n_coord]
            heapq.heappush(boundary, (r_val, n_coord))
            boundary_set.add(n_coord)
            
    # 3. Main Loop
    invaded_resistances = []
    mass_history = []
    rg_history = []
    
    # Tracking coordinates for Rg calculation
    invaded_coords = [np.array(center)]
    
    if max_steps is None:
        max_steps = int(L**dim * 0.4) # Stop before hitting edges too hard
        
    step = 1
    
    while boundary and step < max_steps:
        # Get site with lowest resistance
        r, current_node = heapq.heappop(boundary)
        
        # If already invaded, skip (edge case with duplicates)
        if cluster_grid[current_node] > 0:
            continue
            
        # Invade
        step += 1
        cluster_grid[current_node] = step
        invaded_resistances.append(r)
        invaded_coords.append(np.array(current_node))
        
        # Add neighbors to boundary
        for delta in neighbors:
            n_coord = tuple(np.array(current_node) + np.array(delta))
            
            # Check bounds
            if all(0 <= c < L for c in n_coord):
                # If not invaded and not already in boundary
                if cluster_grid[n_coord] == 0 and n_coord not in boundary_set:
                    r_val = resistances[n_coord]
                    heapq.heappush(boundary, (r_val, n_coord))
                    boundary_set.add(n_coord)
        
        # Calculate Stats periodically (every 100 steps to speed up)
        if step % 100 == 0:
            coords = np.array(invaded_coords)
            M = len(coords)
            # Center of mass
            cm = np.mean(coords, axis=0)
            # Radius of gyration
            rg_sq = np.mean(np.sum((coords - cm)**2, axis=1))
            rg = np.sqrt(rg_sq)
            
            mass_history.append(M)
            rg_history.append(rg)
            
            # Stop if we hit the boundary roughly
            if np.any(coords == 0) or np.any(coords == L-1):
                break

    return cluster_grid, {"M": mass_history, "Rg": rg_history, "r": invaded_resistances}

# ==========================================
# PART 2: Generating Data
# ==========================================
print("Running 2D Simulation (L=400)...")
grid_2d, stats_2d = run_invasion_percolation(L=400, dim=2, max_steps=50000)

print("Running 3D Simulation (L=150)...")
grid_3d, stats_3d = run_invasion_percolation(L=150, dim=3, max_steps=100000)

# ==========================================
# PART 3: Plotting Figure 1 (Scaling Behaviour)
# ==========================================
print("Plotting Figure 1...")
fig1, ax1 = plt.subplots(figsize=(8, 6))

def clean_and_log_data(stats_dict, start_idx=10, max_rg_log=None):
    # Convert to numpy arrays
    rg = np.array(stats_dict["Rg"])
    m = np.array(stats_dict["M"])
    
    # 1. Filter out zeros and NaNs
    valid_mask = (rg > 0) & (m > 0) & (~np.isnan(rg)) & (~np.isnan(m))
    rg = rg[valid_mask]
    m = m[valid_mask]
    
    # 2. Log transformation
    log_rg = np.log10(rg)
    log_m = np.log10(m)
    
    # 3. Skip the noisy start (Cleans data)
    if len(log_rg) > start_idx + 5:
        log_rg = log_rg[start_idx:]
        log_m = log_m[start_idx:]
        
    # 4. Trim the end (Fixing wall effect)
    if max_rg_log is not None:
        mask = log_rg < max_rg_log
        log_rg = log_rg[mask]
        log_m = log_m[mask]
    
    return log_rg, log_m

# --- Process 2D Data ---
try:
    log_rg_2d, log_m_2d = clean_and_log_data(stats_2d, start_idx=20, max_rg_log=2.2)
    
    if len(log_rg_2d) > 2:
        slope_2d, intercept_2d, _, _, _ = linregress(log_rg_2d, log_m_2d)
        
        ax1.scatter(log_rg_2d, log_m_2d, s=20, c='blue', alpha=0.4, label='2D Data')
        
        # Plot the regression line
        fit_y = slope_2d * log_rg_2d + intercept_2d
        ax1.plot(log_rg_2d, fit_y, 'b--', linewidth=2, 
                 label=f'2D Fit: $d_f = {slope_2d:.2f}$')
    else:
        print("Error: Not enough valid 2D points to plot.")

except Exception as e:
    print(f"Could not plot 2D data: {e}")

# --- Process 3D Data ---
try:
    log_rg_3d, log_m_3d = clean_and_log_data(stats_3d, start_idx=5, max_rg_log=1.5)
    
    if len(log_rg_3d) > 2:
        slope_3d, intercept_3d, _, _, _ = linregress(log_rg_3d, log_m_3d)
        
        ax1.scatter(log_rg_3d, log_m_3d, s=20, c='red', alpha=0.4, marker='^', label='3D Data')
        
        # Plot regression line
        fit_y = slope_3d * log_rg_3d + intercept_3d
        ax1.plot(log_rg_3d, fit_y, 'r-.', linewidth=2, 
                 label=f'3D Fit: $d_f = {slope_3d:.2f}$')
    else:
        print("Not enough 3D points for regression.")
except Exception as e:
    print(f"Could not plot 3D data: {e}")

ax1.set_xlabel(r'$\log_{10}(R_g)$', fontsize=14)
ax1.set_ylabel(r'$\log_{10}(M)$', fontsize=14)
ax1.set_title(r'Fractal Dimension Scaling ($M \sim R_g^{d_f}$)', fontsize=16)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Figure1_FractalDimension.png', dpi=300)
plt.show()

# ==========================================
# PART 4: Plotting Figure 2 (SOC Histogram)
# ==========================================
print("Plotting Figure 2...")

fig2, ax2 = plt.subplots(figsize=(8, 4)) 

# Use the 3D resistances for the histogram
r_data = stats_3d["r"]

# Create Histogram
counts, bins, patches = ax2.hist(r_data, bins=50, density=True, color='purple', alpha=0.7, edgecolor='black')

# Add theoretical line
pc_3d = 0.3116 
ax2.axvline(pc_3d, color='red', linestyle='--', linewidth=2, label=f'Theoretical $p_c \\approx {pc_3d}$')

# Labels
ax2.set_xlabel('Resistance Value ($r$)', fontsize=14)
ax2.set_ylabel('Probability Density', fontsize=14)
ax2.set_title('Self-Organized Criticality: Acceptance Profile', fontsize=16)

ax2.legend(fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Figure2_SOC_Histogram.png', dpi=300)
plt.show()

# ==========================================
# PART 5: Plotting Figure 3 (3D Voxel)
# ==========================================
print("Plotting Figure 3...")
fig3 = plt.figure(figsize=(10, 8))
ax3 = fig3.add_subplot(111, projection='3d')

# Crop the grid
invaded_indices = np.argwhere(grid_3d > 0)
min_idx = invaded_indices.min(axis=0)
max_idx = invaded_indices.max(axis=0)
cropped_grid = grid_3d[min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]

# Create boolean mask
filled = cropped_grid > 0

# Color by time
colors = plt.cm.plasma(cropped_grid / np.max(cropped_grid))

# Plot voxels
ax3.voxels(filled, facecolors=colors, edgecolors='k', linewidth=0.1, alpha=0.9)

ax3.set_title('3D Invasion Percolation Cluster\n(Color = Invasion Time)', fontsize=16)

# Add colorbar
m = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
m.set_array(cropped_grid)
cbar = plt.colorbar(m, ax=ax3, shrink=0.5, pad=0.1)
cbar.set_label('Invasion Step', fontsize=12)
cbar.ax.tick_params(labelsize=10) 

plt.tight_layout()
plt.savefig('Figure3_3D_Visualisation.png', dpi=300)
plt.show()

print("Done! All figures saved.")