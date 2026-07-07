






























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
NODE_GRAY = '#bdc3c7'
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
# PANTALLA 3-6: PRIM Y KRUSKAL PASO A PASO (BEST Y WORST)
# =============================================================================

def draw_prim_step(visited, prev_edges, new_edge, step, total_steps, vars_list, G, POS):
    plt.figure(figsize=(14, 10))
    nc = len(vars_list)
    node_colors = [NODE_GREEN if visited[i] else NODE_GRAY for i in range(nc)]
    nx.draw_networkx_nodes(G, POS, node_color=node_colors, node_size=2200,
                           edgecolors=NODE_BORDER, linewidths=2.5)
    nx.draw_networkx_labels(G, POS, {v: v for v in vars_list}, font_size=14,
                            font_weight='bold', font_color='white')
    el = [(vars_list[u], vars_list[v]) for u, v, _ in prev_edges]
    ew = {}
    for u, v, d in prev_edges:
        ew[(vars_list[u], vars_list[v])] = f"{d:.4f}"
    if el:
        nx.draw_networkx_edges(G, POS, edgelist=el, edge_color=EDGE_GREEN, width=3.5, alpha=0.8)
    if new_edge:
        u, v, d = new_edge
        nx.draw_networkx_edges(G, POS, edgelist=[(vars_list[u], vars_list[v])],
                               edge_color=EDGE_NEW, width=6, alpha=0.9)
        ew[(vars_list[u], vars_list[v])] = f"{d:.4f}"
    if ew:
        nx.draw_networkx_edge_labels(G, POS, edge_labels=ew, font_size=9, font_weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    plt.title(f"PRIM MIN - Paso {step}/{total_steps}", fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
    if new_edge:
        u, v, d = new_edge
        plt.suptitle(f"{vars_list[u]} ------ {vars_list[v]}   (d = {d:.4f})", fontsize=14, color=NODE_GREEN, y=0.82)
    plt.figtext(0.5, 0.01, "PRESIONE ENTER en esta ventana para continuar",
                ha='center', fontsize=11, color='#7f8c8d', fontweight='bold')
    plt.axis('off')
    plt.tight_layout()


def draw_kruskal_step(accepted, current_edge, is_accepted, step, connected, n_acc, total, vars_list, G, POS):
    plt.figure(figsize=(14, 10))
    nc = len(vars_list)
    node_colors = [NODE_GREEN if connected[i] else NODE_GRAY for i in range(nc)]
    nx.draw_networkx_nodes(G, POS, node_color=node_colors, node_size=2200,
                           edgecolors=NODE_BORDER, linewidths=2.5)
    nx.draw_networkx_labels(G, POS, {v: v for v in vars_list}, font_size=14,
                            font_weight='bold', font_color='white')
    ew = {}
    if accepted:
        al = [(vars_list[u], vars_list[v]) for u, v, _ in accepted]
        for u, v, d in accepted:
            ew[(vars_list[u], vars_list[v])] = f"{d:.4f}"
        nx.draw_networkx_edges(G, POS, edgelist=al, edge_color=EDGE_GREEN, width=3.5, alpha=0.8)
    u, v, d = current_edge
    cl = (vars_list[u], vars_list[v])
    if is_accepted:
        nx.draw_networkx_edges(G, POS, edgelist=[cl], edge_color=EDGE_NEW, width=6, alpha=0.9)
        ew[cl] = f"{d:.4f}"
    else:
        nx.draw_networkx_edges(G, POS, edgelist=[cl], edge_color=EDGE_RED, width=3, alpha=0.9, style='dashed')
    if ew:
        nx.draw_networkx_edge_labels(G, POS, edge_labels=ew, font_size=9, font_weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    vd = "ACEPTADA" if is_accepted else "RECHAZADA (ciclo)"
    vc = NODE_GREEN if is_accepted else EDGE_RED
    plt.title(f"KRUSKAL MIN - Arista #{step}  [{n_acc}/{total}]", fontsize=18, fontweight='bold', color='#2c3e50', pad=20)
    plt.suptitle(f"{vars_list[u]} ------ {vars_list[v]}   (d = {d:.4f})", fontsize=14, color=vc, y=0.82)
    plt.figtext(0.5, 0.06, vd, ha='center', fontsize=16, fontweight='bold', color=vc)
    plt.figtext(0.5, 0.01, "PRESIONE ENTER en esta ventana para continuar",
                ha='center', fontsize=11, color='#7f8c8d', fontweight='bold')
    plt.axis('off')
    plt.tight_layout()


for name in ["BEST", "WORST"]:
    dd = all_data[name]
    vars_list = dd['vars']
    nv = len(vars_list)
    total_steps = nv - 1

    G = nx.Graph()
    for v in vars_list: G.add_node(v)
    for i in range(nv):
        for j in range(i + 1, nv):
            G.add_edge(vars_list[i], vars_list[j])
    POS = nx.spring_layout(G, seed=42, k=4, iterations=100)

    # Prim MIN paso a paso
    print(f"\n  >>> {name} - PRIM MIN (paso a paso)")
    visited = [False] * nv; visited[0] = True
    prim_edges = []; prim_cost = 0
    draw_prim_step(visited, [], None, 0, total_steps, vars_list, G, POS)
    wait_for_enter()

    for step in range(1, total_steps + 1):
        bd = float('inf'); bu = bv = -1
        for u in range(nv):
            if visited[u]:
                for v in range(nv):
                    if not visited[v] and dd['dist_mat'][u, v] < bd:
                        bd = dd['dist_mat'][u, v]; bu, bv = u, v
        visited[bv] = True
        ne = (bu, bv, bd)
        prim_edges.append(ne); prim_cost += bd
        print(f"    Paso {step}: {vars_list[bu]} -- {vars_list[bv]}  d={bd:.6f}")
        draw_prim_step(visited, prim_edges, ne, step, total_steps, vars_list, G, POS)
        wait_for_enter()

    # Kruskal MIN paso a paso
    print(f"\n  >>> {name} - KRUSKAL MIN (paso a paso)")
    all_edges = []
    for i in range(nv):
        for j in range(i + 1, nv):
            all_edges.append((dd['dist_mat'][i, j], i, j))
    all_edges.sort()
    uf = UnionFind(nv)
    kruskal_edges = []; kruskal_cost = 0; connected = [False] * nv
    step_k = 0

    draw_kruskal_step([], (0, 0, 0), True, 0, connected, 0, total_steps, vars_list, G, POS)
    wait_for_enter()

    for d, u, v in all_edges:
        step_k += 1
        accepted = uf.union(u, v)
        if accepted:
            kruskal_edges.append((u, v, d)); kruskal_cost += d
            connected[u] = True; connected[v] = True
        vd2 = "ACEPTADA" if accepted else "RECHAZADA"
        print(f"    Eval #{step_k}: {vars_list[u]} -- {vars_list[v]}  d={d:.6f}  {vd2}  ({len(kruskal_edges)}/{total_steps})")
        draw_kruskal_step(kruskal_edges, (u, v, d), accepted, step_k, connected, len(kruskal_edges), total_steps, vars_list, G, POS)
        wait_for_enter()
        if accepted and len(kruskal_edges) == total_steps:
            break

# =============================================================================
# PANTALLA 7: ARBOLES INDIVIDUALES (MIN + MAX)
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

# =============================================================================
# PANTALLA 8: SELECCION DE VARIABLES
# =============================================================================

print("\n  >>> Mostrando SELECCION DE VARIABLES...")

# Calcular Delta: metodo rVP - distancias de caminos en el MST
import networkx as nxx

def calc_rVP(mi_mat, vars_list):
    n = len(vars_list)
    G = nxx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=mi_mat[i,j])
    G_neg = nxx.Graph()
    for u,v,d in G.edges(data=True):
        G_neg.add_edge(u, v, weight=-d['weight'])
    mst_neg = nxx.minimum_spanning_tree(G_neg)
    G_mst = nxx.Graph()
    for u,v in mst_neg.edges():
        G_mst.add_edge(vars_list[u], vars_list[v], weight=mi_mat[u,v])
    path_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                p = nxx.shortest_path(G_mst, vars_list[i], vars_list[j], weight='weight')
                path_mat[i,j] = sum(G_mst[p[k]][p[k+1]]['weight'] for k in range(len(p)-1))
    return {i: sum(path_mat[i]) for i in range(n)}

sum_b = calc_rVP(all_data['BEST']['mi_mat'], vars_list)
sum_w = calc_rVP(all_data['WORST']['mi_mat'], vars_list)

ranking = []
for i, var in enumerate(vars_list):
    sb = sum_b[i]; sw = sum_w[i]; delta = sb - sw
    ranking.append((var, sb, sw, delta, abs(delta)))
ranking.sort(key=lambda x: -x[4])

# Gap detection
deltas_rvp = [ad for _, _, _, _, ad in ranking]
gaps_rvp = [deltas_rvp[i]/deltas_rvp[i+1] if deltas_rvp[i+1] > 0 else 999 for i in range(len(deltas_rvp)-1)]
corte_rvp = gaps_rvp.index(max(gaps_rvp))

fig, (ax_bar, ax_table) = plt.subplots(1, 2, figsize=(18, 9),
    gridspec_kw={'width_ratios': [1.2, 1]})
fig.suptitle("METODO 1 (rVP): Distancias de camino en MST", fontsize=16,
             fontweight='bold', color='#2c3e50', y=0.98)

# Barras
vars_rev = [r[0] for r in ranking[::-1]]
deltas_rev = [r[3] for r in ranking[::-1]]
colors = ['#e74c3c' if i <= corte_rvp else '#27ae60' for i, (_, _, _, _, _) in enumerate(ranking[::-1])]
ax_bar.barh(range(len(vars_rev)), deltas_rev, color=colors, edgecolor='white')
for i, (v, d) in enumerate(zip(vars_rev, deltas_rev)):
    ax_bar.text(d + 0.05, i, f"{v} ({d:.4f})", va='center', fontsize=12, fontweight='bold')
ax_bar.set_yticks(range(len(vars_rev)))
ax_bar.set_yticklabels(vars_rev, fontsize=13, fontweight='bold')
ax_bar.set_xlabel('|Delta| = |SUM_BEST - SUM_WORST|', fontsize=11)
ax_bar.set_title(f'Rojo = CRITICA (pos <= {corte_rvp+1}), Verde = estable', fontsize=12,
                 fontweight='bold', color='#2c3e50')

# Tabla
ax_table.axis('off')
tbl_data = [['Var', 'SUM BEST', 'SUM WORST', 'Delta', '|Delta|', 'Critica?']]
for var, sb, sw, d, ad in ranking:
    critica = 'SI' if ad >= deltas_rvp[corte_rvp] else 'No'
    tbl_data.append([var, f'{sb:.4f}', f'{sw:.4f}', f'{d:+.4f}', f'{ad:.4f}', critica])
tbl = ax_table.table(cellText=tbl_data, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.1, 1.6)
for key, cell in tbl.get_celld().items():
    cell.set_edgecolor('#2c3e50'); cell.set_linewidth(1)
    if key[0] == 0: cell.set_facecolor('#2c3e50'); cell.get_text().set_color('white')
ax_table.set_title('Ranking rVP (caida natural)', fontsize=13, fontweight='bold', pad=15)

criticas = [v for i, (v, _, _, _, _) in enumerate(ranking) if i <= corte_rvp]
info2 = f"CRITICAS (caida natural): {criticas}  |  Gap en pos {corte_rvp+1}"
fig.text(0.5, 0.01, info2, ha='center', fontsize=13, color='#27ae60', fontweight='bold')
plt.figtext(0.5, 0.03, "PRESIONE ENTER para finalizar", ha='center',
            fontsize=11, color='#7f8c8d', fontweight='bold')
plt.tight_layout()
wait_for_enter()

# =============================================================================
# PANTALLA 9: METODO PROFESOR (suma fila completa matriz distancias)
# =============================================================================

print("\n  >>> Mostrando METODO 2 (Profesor): Matriz de Distancias...")

sum_pb = {}; sum_pw = {}
for i in range(len(vars_list)):
    sum_pb[i] = sum(all_data['BEST']['dist_mat'][i])
    sum_pw[i] = sum(all_data['WORST']['dist_mat'][i])

ranking_p = []
for i, var in enumerate(vars_list):
    sb = sum_pb[i]; sw = sum_pw[i]; delta = sb - sw
    ranking_p.append((var, sb, sw, delta, abs(delta)))
ranking_p.sort(key=lambda x: -x[4])

# Gap detection
deltas_p = [ad for _, _, _, _, ad in ranking_p]
gaps_p = [deltas_p[i]/deltas_p[i+1] if deltas_p[i+1] > 0 else 999 for i in range(len(deltas_p)-1)]
corte_p = gaps_p.index(max(gaps_p))

fig, (ax_bar, ax_table) = plt.subplots(1, 2, figsize=(18, 9),
    gridspec_kw={'width_ratios': [1.2, 1]})
fig.suptitle("METODO 2 (Profesor): Suma fila completa matriz distancias", fontsize=16,
             fontweight='bold', color='#2c3e50', y=0.98)

vars_rev = [r[0] for r in ranking_p[::-1]]
deltas_rev = [r[3] for r in ranking_p[::-1]]
colors = ['#e74c3c' if i <= corte_p else '#27ae60' for i, (_, _, _, _, _) in enumerate(ranking_p[::-1])]
ax_bar.barh(range(len(vars_rev)), deltas_rev, color=colors, edgecolor='white')
for i, (v, d) in enumerate(zip(vars_rev, deltas_rev)):
    ax_bar.text(d + 0.001, i, f"{v} ({d:+.4f})", va='center', fontsize=12, fontweight='bold')
ax_bar.set_yticks(range(len(vars_rev)))
ax_bar.set_yticklabels(vars_rev, fontsize=13, fontweight='bold')
ax_bar.set_xlabel('Delta = SUM_BEST - SUM_WORST', fontsize=11)
ax_bar.set_title(f'Rojo = discriminativa (pos <= {corte_p+1}), Verde = estable', fontsize=12,
                 fontweight='bold', color='#2c3e50')

ax_table.axis('off')
tbl_data = [['Var', 'SUM BEST', 'SUM WORST', 'Delta', '|Delta|']]
for var, sb, sw, d, ad in ranking_p:
    tbl_data.append([var, f'{sb:.4f}', f'{sw:.4f}', f'{d:+.4f}', f'{ad:.4f}'])
tbl = ax_table.table(cellText=tbl_data, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.1, 1.6)
for key, cell in tbl.get_celld().items():
    cell.set_edgecolor('#2c3e50'); cell.set_linewidth(1)
    if key[0] == 0: cell.set_facecolor('#2c3e50'); cell.get_text().set_color('white')
ax_table.set_title('Ranking por distancias', fontsize=13, fontweight='bold', pad=15)

disc_p = [v for i, (v, _, _, _, _) in enumerate(ranking_p) if i <= corte_p]
info3 = f"Discriminativas (caida natural): {disc_p}  |  Gap en pos {corte_p+1}"
fig.text(0.5, 0.01, info3, ha='center', fontsize=13, color='#e74c3c', fontweight='bold')
plt.figtext(0.5, 0.03, "PRESIONE ENTER para finalizar", ha='center',
            fontsize=11, color='#7f8c8d', fontweight='bold')
plt.tight_layout()
wait_for_enter()

print("\n" + "=" * 70)
print("  VISUALIZACION COMPLETADA")
print("=" * 70)
print()
