"""
=====================================================================
  generar_csvs.py - Genera d9_strong_B.csv y d9_strong_W.csv
  Desde ../d9_strong.txt con 7 columnas (x1,x2,x3,x4,x5,x6,x9)
  Y = x7 (= x8) se usa SOLO para particionar, NO se incluye en los CSVs
  BEST = Y=1 (502)   |   WORST = Y=0 (498)
=====================================================================
"""

print("=" * 60)
print("  GENERADOR DE CSVs: BEST (Y=1) y WORST (Y=0)")
print("=" * 60)

# Leer dataset original
with open("../d9_strong.txt", "r") as f:
    lines = f.readlines()

header = lines[0].strip().split(",")
print(f"\n  Dataset original: {len(lines)-1} filas x {len(header)} variables")

# Reordenar. Y (x7=x8) solo para particionar
rows = []
for line in lines[1:]:
    vals = [int(x) for x in line.strip().split(",")]
    x1 = vals[0]
    x2 = vals[1]
    x3 = vals[6]
    x4 = vals[3]
    x5 = vals[2]
    x6 = vals[4]
    Y  = vals[7]  # x7 = x8 = particion
    x9 = vals[5]
    rows.append((x1, x2, x3, x4, x5, x6, x9, Y))  # Y al final para filtrar

# Separar por Y. Guardar SOLO x1-x6,x9 (7 columnas, sin Y)
best = [r[:7] for r in rows if r[7] == 1]
worst = [r[:7] for r in rows if r[7] == 0]

# Guardar BEST
with open("d9_strong_B.csv", "w") as f:
    f.write("x1,x2,x3,x4,x5,x6,x9\n")
    for r in best:
        f.write(",".join(str(v) for v in r) + "\n")

# Guardar WORST
with open("d9_strong_W.csv", "w") as f:
    f.write("x1,x2,x3,x4,x5,x6,x9\n")
    for r in worst:
        f.write(",".join(str(v) for v in r) + "\n")

print(f"\n  CSVs generados (7 columnas, sin Y):")
print(f"    d9_strong_B.csv  ->  {len(best)} filas (Y=1, BEST)")
print(f"    d9_strong_W.csv  ->  {len(worst)} filas (Y=0, WORST)")
print(f"    Total:            {len(best) + len(worst)} filas")
print(f"\n  NOTA: Y se uso solo para particionar.")
print(f"  Los CSVs contienen 7 variables: x1, x2, x3, x4, x5, x6, x9")
print()
