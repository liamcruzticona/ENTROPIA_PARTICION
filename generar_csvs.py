"""
=====================================================================
  generar_csvs.py - Genera d9_strong_B.csv y d9_strong_W.csv
  Desde ../d9_strong.txt con 8 columnas (x7+x8 combinadas en Y)
  BEST = Y=1 (502)   |   WORST = Y=0 (498)
=====================================================================
"""

import os

print("=" * 60)
print("  GENERADOR DE CSVs: BEST (Y=1) y WORST (Y=0)")
print("=" * 60)

# Leer dataset original
with open("../d9_strong.txt", "r") as f:
    lines = f.readlines()

header = lines[0].strip().split(",")
print(f"\n  Dataset original: {len(lines)-1} filas x {len(header)} variables")

# Reordenar y combinar x7+x8 en Y
rows = []
for line in lines[1:]:
    vals = [int(x) for x in line.strip().split(",")]
    x1 = vals[0]  # x1
    x2 = vals[1]  # x2
    x3 = vals[6]  # x3
    x4 = vals[3]  # x4
    x5 = vals[2]  # x5
    x6 = vals[4]  # x6
    Y  = vals[8]  # x7 (= x8, usamos x7)
    x9 = vals[5]  # x9
    rows.append([x1, x2, x3, x4, x5, x6, Y, x9])

# Separar por Y
best = [r for r in rows if r[6] == 1]
worst = [r for r in rows if r[6] == 0]

# Guardar BEST
with open("d9_strong_B.csv", "w") as f:
    f.write("x1,x2,x3,x4,x5,x6,Y,x9\n")
    for r in best:
        f.write(",".join(str(v) for v in r) + "\n")

# Guardar WORST
with open("d9_strong_W.csv", "w") as f:
    f.write("x1,x2,x3,x4,x5,x6,Y,x9\n")
    for r in worst:
        f.write(",".join(str(v) for v in r) + "\n")

print(f"\n  CSVs generados:")
print(f"    d9_strong_B.csv  ->  {len(best)} filas (Y=1, BEST)")
print(f"    d9_strong_W.csv  ->  {len(worst)} filas (Y=0, WORST)")
print(f"    Total:            {len(best) + len(worst)} filas")
print(f"\n  Columnas (8): x1, x2, x3, x4, x5, x6, Y, x9")
print()
