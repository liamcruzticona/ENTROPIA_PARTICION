"""
=====================================================================
  generar_reducido.py - Genera d9_reducido.csv con variables seleccionadas
  Toma el ranking del arbol MAX (BEST vs WORST) y genera dataset reducido
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

    mi_dict = {}
    for v1, v2 in combinations(range(n_cols), 2):
        m = mutual_info(d[:, v1], d[:, v2], n_rows)
        mi_dict[(v1, v2)] = m; mi_dict[(v2, v1)] = m

    mi_matrix = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j: mi_matrix[i, j] = mi_dict.get((i, j), 0.0)

    dist_matrix = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            if i != j:
                min_h = min(entropies[variables[i]], entropies[variables[j]])
                dist_matrix[i, j] = max(0.0, 1.0 - mi_matrix[i, j] / min_h) if min_h > 0 else 0.0

    # Suma de fila completa de la matriz de distancias
    suma = {}
    for i in range(n_cols):
        suma[i] = sum(dist_matrix[i])

    return variables, dist_matrix, suma


# =============================================================================
r_b = pipeline('d9_strong_B.csv')
r_w = pipeline('d9_strong_W.csv')

ranking = []
for i, var in enumerate(r_b[0]):
    sb = r_b[2][i]
    sw = r_w[2][i]
    delta = sb - sw
    ranking.append((var, sb, sw, delta, i))

ranking.sort(key=lambda x: -abs(x[3]))

print("=" * 60)
print("  SELECCION DE VARIABLES - RANKING")
print("=" * 60)
print(f"\n  {'Variable':<10} {'SUM BEST':<14} {'SUM WORST':<14} {'Delta':<12}")
for var, sb, sw, d, _ in ranking:
    print(f"  {var:<10} {sb:<14.6f} {sw:<14.6f} {d:<12.6f}")

# Caida natural: top 4
keep_idx = [idx for _, _, _, _, idx in ranking[:4]]
keep_vars = [v for v, _, _, _, _ in ranking[:4]]

print(f"\n  Top 4 variables (caida natural del Delta):")
print(f"  {keep_vars}")
print(f"  (Delta cae de {ranking[3][3]:.6f} a {ranking[4][3]:.6f})")

# Cargar dataset original completo (1000x9)
with open('../d9_strong.txt') as f:
    lines = f.readlines()
header = lines[0].strip().split(',')

# Reordenar a x1,x2,x3,x4,x5,x6,x7,x9 (usando x8=x7)
var_map = {'x1': 0, 'x2': 1, 'x3': 6, 'x4': 3, 'x5': 2, 'x6': 4, 'x7': 7, 'x9': 5}

# Construir dataset reducido
reduced_rows = []
for line in lines[1:]:
    vals = [int(x) for x in line.strip().split(',')]
    row = []
    for var in keep_vars:
        idx = var_map[var]
        row.append(vals[idx])
    reduced_rows.append(row)

# Guardar
output = f'd9_reducido_K{len(keep_vars)}.csv'
with open(output, 'w') as f:
    f.write(','.join(keep_vars) + '\n')
    for row in reduced_rows:
        f.write(','.join(str(v) for v in row) + '\n')

print(f"\n  Dataset reducido generado: {output}")
print(f"  Dimension: {len(reduced_rows)} filas x {len(keep_vars)} columnas")
print(f"  Variables: {keep_vars}")
print()
