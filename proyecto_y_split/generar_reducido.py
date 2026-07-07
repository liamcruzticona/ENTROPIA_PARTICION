"""
=====================================================================
  generar_reducido.py - Metodo rVP (Rango de Variacion Permitido)
  Distancias de caminos dentro del MST (MAX, MI)
=====================================================================
"""

import numpy as np
import networkx as nx
from collections import Counter
from itertools import combinations


def entropy(probs):
    return -sum(p * np.log2(p) for p in probs if p > 0)


def mutual_info(col_x, col_y, n):
    jc = {}
    for i in range(n): k = (col_x[i], col_y[i]); jc[k] = jc.get(k, 0) + 1
    mi = 0.0
    for (vx, vy), c in jc.items():
        p_xy = c / n; p_x = np.sum(col_x == vx) / n; p_y = np.sum(col_y == vy) / n
        mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi


def rVP(mi_matrix, variables):
    n = len(variables)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=mi_matrix[i, j])
    G_neg = nx.Graph()
    for u, v, data in G.edges(data=True):
        G_neg.add_edge(u, v, weight=-data['weight'])
    mst_neg = nx.minimum_spanning_tree(G_neg)
    G_mst = nx.Graph()
    for u, v in mst_neg.edges():
        G_mst.add_edge(variables[u], variables[v], weight=mi_matrix[u, v])
    path = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p = nx.shortest_path(G_mst, variables[i], variables[j], weight='weight')
                path[i, j] = sum(G_mst[p[k]][p[k+1]]['weight'] for k in range(len(p)-1))
    return {i: sum(path[i]) for i in range(n)}


def pipeline(filename):
    with open(filename) as f: lines = f.readlines()
    h = lines[0].strip().split(',')
    d = np.array([[int(x) for x in l.strip().split(',')] for l in lines[1:]])
    n_rows, n_cols = len(d), len(h)
    variables = h
    prob_tables = {}
    for j, var in enumerate(variables):
        col = d[:, j]; counts = Counter(col)
        prob_tables[var] = {k: v / n_rows for k, v in sorted(counts.items())}
    entropies = {var: entropy(list(prob_tables[var].values())) for var in variables}
    mi_matrix = np.zeros((n_cols, n_cols))
    for v1, v2 in combinations(range(n_cols), 2):
        m = mutual_info(d[:, v1], d[:, v2], n_rows)
        mi_matrix[v1, v2] = m; mi_matrix[v2, v1] = m
    return variables, mi_matrix


# =============================================================================
r_b = pipeline('d9_strong_B.csv')
r_w = pipeline('d9_strong_W.csv')

sum_b = rVP(r_b[1], r_b[0])
sum_w = rVP(r_w[1], r_w[0])

ranking = []
for i, var in enumerate(r_b[0]):
    sb = sum_b[i]; sw = sum_w[i]; delta = sb - sw
    ranking.append((var, sb, sw, delta, abs(delta)))

ranking.sort(key=lambda x: -x[4])

umbral = 0.5

print("=" * 60)
print("  SELECCION DE VARIABLES - Metodo rVP")
print("=" * 60)
print(f"  {'Variable':<10} {'SUM BEST':<14} {'SUM WORST':<14} {'Delta':>12} {'|Delta|':<10} {'Critica?'}")
print(f"  {'-'*68}")
for var, sb, sw, d, ad in ranking:
    critica = 'SI' if ad > umbral else 'No'
    print(f"  {var:<10} {sb:<14.4f} {sw:<14.4f} {d:12.4f} {ad:<10.4f} {critica}")

criticas = [v for v, _, _, _, ad in ranking if ad > umbral]
estables = [v for v, _, _, _, ad in ranking if ad <= umbral]
print(f"\n  CRITICAS (|Delta| > {umbral}): {criticas}")
print(f"  ESTABLES: {estables}")
print()
