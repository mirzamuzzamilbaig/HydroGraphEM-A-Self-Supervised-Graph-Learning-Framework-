import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import random

random.seed(42)
np.random.seed(42)

# Build a dendritic tree with exactly 483 nodes
G = nx.DiGraph()
G_undirected = nx.Graph()
node_id = 0
G.add_node(node_id)
G_undirected.add_node(node_id)
depth_map = {0: 0}
queue = [node_id]
node_id += 1
target_nodes = 483

while node_id < target_nodes and queue:
    parent = queue.pop(0)
    d = depth_map[parent]
    remaining = target_nodes - node_id
    queue_size = len(queue)

    # Aggressive branching to reach 483
    if d < 2:
        n_children = random.choice([3, 3, 4])
    elif d < 4:
        n_children = random.choice([2, 2, 3, 3])
    elif d < 7:
        n_children = random.choice([1, 2, 2, 3])
    elif d < 10:
        n_children = random.choice([1, 1, 2, 2])
    elif d < 14:
        n_children = random.choice([0, 1, 1, 2])
    else:
        n_children = random.choice([0, 0, 1])

    for _ in range(n_children):
        if node_id >= target_nodes:
            break
        G.add_node(node_id)
        G.add_edge(node_id, parent)
        G_undirected.add_edge(node_id, parent)
        depth_map[node_id] = d + 1
        queue.append(node_id)
        node_id += 1

actual_nodes = len(G.nodes())
print(f"Built tree with {actual_nodes} nodes, {len(G.edges())} edges")

# Compute Strahler stream order
strahler = {}
def compute_strahler(node):
    preds = list(G.predecessors(node))
    if not preds:
        strahler[node] = 1
        return 1
    child_orders = []
    for child in preds:
        if child not in strahler:
            compute_strahler(child)
        child_orders.append(strahler[child])
    child_orders.sort(reverse=True)
    if len(child_orders) == 1:
        strahler[node] = child_orders[0]
    elif child_orders[0] == child_orders[1]:
        strahler[node] = child_orders[0] + 1
    else:
        strahler[node] = child_orders[0]
    return strahler[node]

import sys
sys.setrecursionlimit(5000)
compute_strahler(0)

for n in strahler:
    strahler[n] = min(strahler[n], 5)

# Layout
pos = nx.spring_layout(G_undirected, k=2.0/np.sqrt(actual_nodes), iterations=100, seed=42)

# Flip y so outlet at bottom
ys = [pos[n][1] for n in pos]
max_y, min_y = max(ys), min(ys)
for n in pos:
    pos[n] = (pos[n][0], max_y - pos[n][1] + min_y)

order_colors = {1: '#AED6F1', 2: '#5DADE2', 3: '#2E86C1', 4: '#1B4F72', 5: '#0B2447'}
order_sizes = {1: 8, 2: 14, 3: 24, 4: 40, 5: 65}

# Select ~105 holdout nodes
holdout_nodes = set()
for candidate in range(1, actual_nodes):
    ancestors = nx.ancestors(G, candidate)
    subtree = ancestors | {candidate}
    if 95 <= len(subtree) <= 115:
        holdout_nodes = subtree
        break
if len(holdout_nodes) == 0:
    leaves = [n for n in G.nodes() if G.in_degree(n) == 0]
    holdout_nodes = set(leaves[:105])

print(f"Holdout: {len(holdout_nodes)} nodes, Max Strahler: {max(strahler.values())}")

# Plot
fig, ax = plt.subplots(figsize=(8, 10), dpi=300)

for u, v in G.edges():
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color='#BBBBBB', linewidth=0.25, alpha=0.5, zorder=1)

for order in range(1, 6):
    nodes_reg = [n for n in G.nodes() if strahler.get(n, 1) == order and n not in holdout_nodes]
    if nodes_reg:
        ax.scatter([pos[n][0] for n in nodes_reg], [pos[n][1] for n in nodes_reg],
                   s=order_sizes[order], c=order_colors[order],
                   edgecolors='#888888', linewidths=0.15, zorder=3, alpha=0.9)

for order in range(1, 6):
    nodes_ho = [n for n in holdout_nodes if strahler.get(n, 1) == order]
    if nodes_ho:
        ax.scatter([pos[n][0] for n in nodes_ho], [pos[n][1] for n in nodes_ho],
                   s=order_sizes[order]*1.5, c=order_colors[order],
                   edgecolors='#E74C3C', linewidths=0.9, zorder=4, alpha=0.95)

ax.scatter([pos[0][0]], [pos[0][1]], s=130, c='#E74C3C', marker='^',
           edgecolors='black', linewidths=1.0, zorder=5)

legend_handles = [mpatches.Patch(color=order_colors[o], label=f'Stream Order {o}') for o in range(1, 6)]
legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#E74C3C',
                                  markersize=10, markeredgecolor='black', label='Outlet'))
legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#5DADE2',
                                  markersize=8, markeredgecolor='#E74C3C', markeredgewidth=1.5,
                                  label=f'Spatial holdout ({len(holdout_nodes)} reaches)'))
ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#CCCCCC')

stats_text = (f"N = {actual_nodes} reaches\nE = {len(G.edges())} edges\n"
              f"Max Strahler Order = {max(strahler.values())}\n"
              f"Spatial holdout = {len(holdout_nodes)} reaches")
ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))

ax.axis('off')
plt.tight_layout()
plt.savefig('d:/all in one/Desktop/new_PhD_project/prism/network_topology.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("SUCCESS: network_topology.png saved")
