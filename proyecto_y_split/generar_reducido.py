"""
=====================================================================
  generar_reducido.py - Genera d9_reducido.csv con variables seleccionadas
  Metodo: Suma de aristas incidentes en arbol MIN (distancias)
  Ranking por |Delta| = |SUM_BEST - SUM_WORST| con umbral
=====================================================================
"""

import numpy as np
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

    mi_dict = {}; mi_matrix = np.zeros((n_cols, n_cols))
    for v1, v2 in combinations(range(n_cols), 2):
        m = mutual_info(d[:, v1], d[:, v2], n_rows)
        mi_dict[(v1, v2)] = m; mi_dict[(v2, v1)] = m
        mi_matrix[v1, v2] = m; mi_matrix[v2, v1] = m

    dist_matrix = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j:
                min_h = min(entropies[variables[i]], entropies[variables[j]])
                dist_matrix[i, j] = max(0.0, 1.0 - mi_matrix[i, j] / min_h) if min_h > 0 else 0.0

    # Prim MIN
    visited = [False] * n_cols; visited[0] = True
    pmin = []; pmin_cost = 0
    for _ in range(n_cols - 1):
        bd = float('inf'); bu = bv = -1
        for u in range(n_cols):
            if visited[u]:
                for v in range(n_cols):
                    if not visited[v] and dist_matrix[u, v] < bd:
                        bd = dist_matrix[u, v]; bu, bv = u, v
        visited[bv] = True; pmin.append((bu, bv, bd)); pmin_cost += bd

    # Suma pesos de distancias incidentes en arbol MIN
    suma = {}
    for u, v, d in pmin:
        suma[u] = suma.get(u, 0) + d
        suma[v] = suma.get(v, 0) + d

    return variables, pmin, pmin_cost, suma


# =============================================================================
r_b = pipeline('d9_strong_B.csv')
r_w = pipeline('d9_strong_W.csv')

ranking = []
for i, var in enumerate(r_b[0]):
    sb = r_b[3].get(i, 0)
    sw = r_w[3].get(i, 0)
    delta = abs(sb - sw)
    ranking.append((var, sb, sw, delta, i))

ranking.sort(key=lambda x: -x[3])

umbral = 0.01

print("=" * 60)
print("  SELECCION DE VARIABLES - Arbol MIN (distancias)")
print("=" * 60)
print(f"\n  {'Variable':<10} {'SUM BEST':<14} {'SUM WORST':<14} {'|Delta|':<12} {'Decision'}")
print(f"  {'-'*60}")
for var, sb, sw, d, _ in ranking:
    dec = "CONSERVAR" if d >= umbral else "ELIMINAR"
    print(f"  {var:<10} {sb:<14.6f} {sw:<14.6f} {d:<12.6f} {dec}")

keep = [v for v, _, _, d, _ in ranking if d >= umbral]
discard = [v for v, _, _, d, _ in ranking if d < umbral]
keep_idx = [idx for _, _, _, d, idx in ranking if d >= umbral]

print(f"\n  CONSERVAR ({len(keep)}/{len(ranking)}): {keep}")
print(f"  ELIMINAR ({len(discard)}/{len(ranking)}): {discard}")
print(f"  (umbral: |Delta| >= {umbral})")

# Cargar dataset original
with open('../d9_strong.txt') as f:
    lines = f.readlines()
header = lines[0].strip().split(',')

var_map = {'x1': 0, 'x2': 1, 'x3': 6, 'x4': 3, 'x5': 2, 'x6': 4, 'x7': 7, 'x9': 5}

reduced_rows = []
for line in lines[1:]:
    vals = [int(x) for x in line.strip().split(',')]
    row = [vals[var_map[v]] for v in keep]
    reduced_rows.append(row)

output = f'd9_reducido_K{len(keep)}.csv'
with open(output, 'w') as f:
    f.write(','.join(keep) + '\n')
    for row in reduced_rows:
        f.write(','.join(str(v) for v in row) + '\n')

print(f"\n  Dataset reducido: {output}")
print(f"  Dimension: {len(reduced_rows)} filas x {len(keep)} columnas")
print(f"  Variables: {keep}")
print()
