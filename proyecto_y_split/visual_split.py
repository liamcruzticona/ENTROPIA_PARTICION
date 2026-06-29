"""
=====================================================================
  visual_split.py - Arboles BEST vs WORST lado a lado
  Muestra 4 arboles: BEST MIN, BEST MAX, WORST MIN, WORST MAX
  Presione ENTER en la VENTANA GRAFICA para avanzar
=====================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations

plt.ion()

NODE_GREEN = '#2ecc71'
NODE_BLUE = '#3498db'
EDGE_GREEN = '#27ae60'
EDGE_NEW = '#3498db'
EDGE_ORANGE = '#f39c12'
EDGE_RED = '#e74c3c'
NODE_BORDER = '#2c3e50'


import time

def wait_for_enter():
    fig = plt.gcf()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    pressed = []
    def on_key(event):
        if event.key == 'enter':
            pressed.append(True)
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    t0 = time.time()
    while not pressed and (time.time() - t0) < 120:
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)


def entropy(probs):
    return -sum(p * np.log2(p) for p in probs if p > 0)


def mutual_info(col_x, col_y, n):
    jc = {}
    for i in range(n):
        k = (col_x[i], col_y[i])
        jc[k] = jc.get(k, 0) + 1
    mi = 0.0
    for (vx, vy), c in jc.items():
        p_xy = c / n
        p_x = np.sum(col_x == vx) / n
        p_y = np.sum(col_y == vy) / n
        mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True


def run_pipeline(header, data, label):
    n_rows, n_cols = len(data), len(header)
    variables = header

    prob_tables = {}
    for j, var in enumerate(variables):
        col = data[:, j]
        counts = Counter(col)
        prob_tables[var] = {k: v / n_rows for k, v in sorted(counts.items())}

    entropies = {var: entropy(list(prob_tables[var].values())) for var in variables}

    mi_dict = {}
    mi_matrix = np.zeros((n_cols, n_cols))
    for v1, v2 in combinations(range(n_cols), 2):
        m = mutual_info(data[:, v1], data[:, v2], n_rows)
        mi_dict[(v1, v2)] = m
        mi_dict[(v2, v1)] = m
        mi_matrix[v1, v2] = m
        mi_matrix[v2, v1] = m

    dist_matrix = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j:
                min_h = min(entropies[variables[i]], entropies[variables[j]])
                dist_matrix[i, j] = max(0.0, 1.0 - mi_matrix[i, j] / min_h) if min_h > 0 else 0.0

    # Prim MIN
    visited = [False] * n_cols
    visited[0] = True
    prim_min = []
    prim_min_cost = 0.0
    for _ in range(n_cols - 1):
        bd = float('inf')
        bu, bv = -1, -1
        for u in range(n_cols):
            if visited[u]:
                for v in range(n_cols):
                    if not visited[v] and dist_matrix[u, v] < bd:
                        bd = dist_matrix[u, v]
                        bu, bv = u, v
        visited[bv] = True
        prim_min.append((bu, bv, bd))
        prim_min_cost += bd

    # Prim MAX
    visited = [False] * n_cols
    visited[0] = True
    prim_max = []
    prim_max_cost = 0.0
    for _ in range(n_cols - 1):
        bw = -1.0
        bu, bv = -1, -1
        for u in range(n_cols):
            if visited[u]:
                for v in range(n_cols):
                    if not visited[v] and mi_matrix[u, v] > bw:
                        bw = mi_matrix[u, v]
                        bu, bv = u, v
        visited[bv] = True
        prim_max.append((bu, bv, bw))
        prim_max_cost += bw

    return variables, prim_min, prim_min_cost, prim_max, prim_max_cost, mi_matrix, dist_matrix, entropies


# =============================================================================
# CARGA Y CALCULOS COMPLETOS
# =============================================================================

print("\n" + "=" * 70)
print("  VISUAL: MST BEST (Y=1) vs WORST (Y=0)")
print("=" * 70)
print("  Presione ENTER en la VENTANA GRAFICA para avanzar.")

all_data = {}

for name, file in [("BEST", "d9_strong_B.csv"), ("WORST", "d9_strong_W.csv")]:
    with open(file) as f:
        lines = f.readlines()
    h = lines[0].strip().split(',')
    d = np.array([[int(x) for x in l.strip().split(',')] for l in lines[1:]])
    vars_list, pmin, pc_min, pmax, pc_max, mi_mat, dist_mat, ent = run_pipeline(h, d, name)
    all_data[name] = {'vars': vars_list, 'pmin': pmin, 'pc_min': pc_min,
                       'pmax': pmax, 'pc_max': pc_max,
                       'mi_mat': mi_mat, 'dist_mat': dist_mat, 'ent': ent}

# =============================================================================
# PANTALLA 1: MATRICES IM Y DISTANCIAS - BEST
# =============================================================================

for name in ["BEST", "WORST"]:
    dd = all_data[name]
    vars_list = dd['vars']
    mi_mat = dd['mi_mat']
    dist_mat = dd['dist_mat']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"MATRICES: {name} (Y={'1' if name=='BEST' else '0'})",
                 fontsize=16, fontweight='bold', color='#2c3e50', y=0.98)

    # Matriz IM
    vmax = np.max(mi_mat) * 1.05 if np.max(mi_mat) > 0 else 1
    im1 = ax1.imshow(mi_mat, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='equal')
    ax1.set_xticks(range(len(vars_list)))
    ax1.set_xticklabels(vars_list, fontsize=10, fontweight='bold')
    ax1.set_yticks(range(len(vars_list)))
    ax1.set_yticklabels(vars_list, fontsize=10, fontweight='bold')
    for i in range(len(vars_list)):
        for j in range(len(vars_list)):
            val = mi_mat[i, j]
            c = 'white' if val > vmax * 0.35 else '#2c3e50'
            ax1.text(j, i, f"{val:.4f}", ha='center', va='center',
                     fontsize=9, fontweight='bold', color=c)
    ax1.set_title("Matriz IM I(X;Y)\n(Informacion Mutua)", fontsize=12,
                  fontweight='bold', color='#e74c3c')
    fig.colorbar(im1, ax=ax1, shrink=0.82)

    # Matriz Distancias
    im2 = ax2.imshow(dist_mat, cmap='YlGnBu_r', vmin=0, vmax=1, aspect='equal')
    ax2.set_xticks(range(len(vars_list)))
    ax2.set_xticklabels(vars_list, fontsize=10, fontweight='bold')
    ax2.set_yticks(range(len(vars_list)))
    ax2.set_yticklabels(vars_list, fontsize=10, fontweight='bold')
    for i in range(len(vars_list)):
        for j in range(len(vars_list)):
            val = dist_mat[i, j]
            c = 'white' if val > 0.5 else '#2c3e50'
            ax2.text(j, i, f"{val:.4f}", ha='center', va='center',
                     fontsize=9, fontweight='bold', color=c)
    ax2.set_title("Matriz de Distancias d(X;Y)\n(0 = max dependencia)", fontsize=12,
                  fontweight='bold', color='#3498db')
    fig.colorbar(im2, ax=ax2, shrink=0.82)

    plt.figtext(0.5, 0.01, "PRESIONE ENTER en esta ventana para continuar",
                ha='center', fontsize=11, color='#7f8c8d', fontweight='bold')
    plt.tight_layout()
    wait_for_enter()

# =============================================================================
# PANTALLA 2-3: ARBOLES INDIVIDUALES
# =============================================================================

for name in ["BEST", "WORST"]:
    dd = all_data[name]
    vars_list = dd['vars']
    pmin = dd['pmin']
    pc_min = dd['pc_min']
    pmax = dd['pmax']
    pc_max = dd['pc_max']

    # Grafo base
    G = nx.Graph()
    for v in vars_list:
        G.add_node(v)
    for i in range(len(vars_list)):
        for j in range(i + 1, len(vars_list)):
            G.add_edge(vars_list[i], vars_list[j])
    POS = nx.spring_layout(G, seed=42, k=4, iterations=100)

    node_color = NODE_GREEN if name == "BEST" else NODE_BLUE

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(f"MST: {name} (Y={'1' if name=='BEST' else '0'})  |  "
                 f"MIN (distancias) vs MAX (MI directa)",
                 fontsize=16, fontweight='bold', color='#2c3e50', y=0.98)

    for ax, edges, cost, title, ecolor in [
        (ax1, pmin, pc_min, f"{name} MIN (distancias)", EDGE_GREEN),
        (ax2, pmax, pc_max, f"{name} MAX (MI directa)", EDGE_NEW),
    ]:
        nc = [node_color] * len(vars_list)
        nx.draw_networkx_nodes(G, POS, ax=ax, node_color=nc, node_size=2000,
                               edgecolors=NODE_BORDER, linewidths=2)
        nx.draw_networkx_labels(G, POS, ax=ax, labels={v: v for v in vars_list},
                                font_size=14, font_weight='bold', font_color='white')
        el = [(vars_list[u], vars_list[v]) for u, v, _ in edges]
        ew = {(vars_list[u], vars_list[v]): f"{d:.4f}" for u, v, d in edges}
        nx.draw_networkx_edges(G, POS, ax=ax, edgelist=el, edge_color=ecolor,
                               width=4, alpha=0.9)
        nx.draw_networkx_edge_labels(G, POS, ax=ax, edge_labels=ew, font_size=9,
                                      font_weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.2',
                                                facecolor='white', alpha=0.85))
        ax.set_title(f"{title}\nCosto = {cost:.6f}", fontsize=14,
                     fontweight='bold', color=ecolor)
        ax.axis('off')

    plt.figtext(0.5, 0.01, "PRESIONE ENTER en esta ventana para continuar",
                ha='center', fontsize=11, color='#7f8c8d', fontweight='bold')
    plt.tight_layout()
    wait_for_enter()

# =============================================================================
# COMPARACION FINAL - 4 ARBOLES
# =============================================================================

print("\n  >>> Mostrando COMPARACION FINAL: BEST vs WORST (4 arboles)...")

plt.close('all')
plt.pause(0.3)

all_results = all_data

v0 = all_data['BEST']['vars']
G = nx.Graph()
for v in v0:
    G.add_node(v)
for i in range(len(v0)):
    for j in range(i + 1, len(v0)):
        G.add_edge(v0[i], v0[j])
POS = nx.spring_layout(G, seed=42, k=4, iterations=100)

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("COMPARACION FINAL: BEST (Y=1, verde) vs WORST (Y=0, azul)",
             fontsize=18, fontweight='bold', color='#2c3e50', y=0.98)

panels = [
    (axes[0, 0], 'BEST', 'pmin', 'pc_min', EDGE_GREEN, NODE_GREEN,
     "BEST MIN (distancias)"),
    (axes[0, 1], 'BEST', 'pmax', 'pc_max', EDGE_NEW, NODE_GREEN,
     "BEST MAX (MI directa)"),
    (axes[1, 0], 'WORST', 'pmin', 'pc_min', EDGE_ORANGE, NODE_BLUE,
     "WORST MIN (distancias)"),
    (axes[1, 1], 'WORST', 'pmax', 'pc_max', EDGE_RED, NODE_BLUE,
     "WORST MAX (MI directa)"),
]

for ax, name, ekey, ckey, ecolor, ncolor, title in panels:
    r = all_results[name]
    edges = r[ekey]
    cost = r[ckey]
    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=[ncolor] * len(v0),
                           node_size=1800, edgecolors=NODE_BORDER, linewidths=2)
    nx.draw_networkx_labels(G, POS, ax=ax, labels={v: v for v in v0},
                            font_size=13, font_weight='bold', font_color='white')
    el = [(r['vars'][u], r['vars'][v]) for u, v, _ in edges]
    ew = {(r['vars'][u], r['vars'][v]): f"{d:.4f}" for u, v, d in edges}
    nx.draw_networkx_edges(G, POS, ax=ax, edgelist=el, edge_color=ecolor,
                           width=3.5, alpha=0.9)
    nx.draw_networkx_edge_labels(G, POS, ax=ax, edge_labels=ew, font_size=8,
                                  font_weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor='white', alpha=0.85))
    ax.set_title(f"{title}\nCosto = {cost:.6f}", fontsize=14,
                 fontweight='bold', color=ecolor)
    ax.axis('off')

# Calcular aristas en comun
eb = {(min(u, v), max(u, v)) for u, v, _ in all_results['BEST']['pmax']}
ew2 = {(min(u, v), max(u, v)) for u, v, _ in all_results['WORST']['pmax']}
comun = len(eb & ew2)

info = (f"MAX: BEST={all_results['BEST']['pc_max']:.4f}  "
        f"WORST={all_results['WORST']['pc_max']:.4f}  |  "
        f"Aristas en comun MAX: {comun}/6 ({100*comun/6:.0f}%)")
fig.text(0.5, 0.01, info, ha='center', fontsize=13, color='#2c3e50',
         fontweight='bold')

plt.figtext(0.5, 0.02, "PRESIONE ENTER para finalizar",
            ha='center', fontsize=11, color='#7f8c8d', fontweight='bold')
plt.tight_layout()
wait_for_enter()

print("\n" + "=" * 70)
print("  VISUALIZACION COMPLETADA")
print("=" * 70)
print()
