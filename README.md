# ENTROPIA PARTICION

## Analisis Comparativo de la Topologia del Arbol de Expansion entre los Grupos BEST y WORST usando Informacion Mutua y Algoritmos de Prim-Kruskal

**Autor:** Huanacuni Ccama Jhon Edy  
**Universidad:** Universidad Nacional del Altiplano  
**Escuela:** Ingenieria de Sistemas  
**Curso:** Lenguajes de Programacion  
**Docente:** ING. JOSE LUIS JUAREZ RUELAS  
**Semestre:** 2026-I

---

## Descripcion

Pipeline completo de teoria de la informacion y teoria de grafos aplicado al dataset `d9_strong.txt` (1000 registros x 9 variables categoricas). El proyecto implementa la secuencia funcional:

```
Datos → Frecuencias → Entropia H(X) → Informacion Mutua I(X;Y)
→ Matriz IM → Conversión a Distancias → MST (Prim + Kruskal, MIN y MAX)
→ Split BEST vs WORST → Comparacion Topologica
```

---

## Estructura del Repositorio

```
ENTROPIA_PARTICION/
├── README.md
├── .gitignore
├── generar_csvs.py                       ← Genera CSVs desde d9_strong.txt
├── pipeline_comparar.py                  ← Pipeline MIN+MAX en BEST y WORST
├── visual_split.py                       ← Visual interactiva BEST vs WORST
├── d9_strong_B.csv                       ← BEST: 502 filas (Y=1)
├── d9_strong_W.csv                       ← WORST: 498 filas (Y=0)
├── resultados_split.xlsx                 ← Excel con 3 hojas
└── informe_apa7_split.tex                ← Informe APA 7 (definitivo)
```

---

## Requisitos

| Libreria | Version | Uso |
|----------|---------|-----|
| Python | 3.x | Lenguaje base |
| numpy | - | Calculos numericos |
| matplotlib | - | Visualizacion de grafos |
| networkx | - | Construccion y dibujo de arboles |
| openpyxl | - | Generacion de archivos Excel |

---

## Como Ejecutar

```powershell
cd ENTROPIA_PARTICION

# Generar CSVs (si no existen)
python generar_csvs.py

# Pipeline comparativo + Excel
python pipeline_comparar.py

# Visualizacion interactiva (matrices + arboles)
python visual_split.py
```

---

## Resultados Principales

### Split BEST vs WORST

| Metrica | BEST (Y=1) | WORST (Y=0) |
|---------|-----------|------------|
| Registros | 502 | 498 |
| H(Y) | 0 (constante) | 0 (constante) |
| Hub MAX (MI directa) | x1 (grado 3) | x9 (grado 4) |
| Costo MIN (distancias) | 0.000000 | 0.000000 |
| Costo MAX (MI directa) | 2.833722 | 2.809188 |
| Aristas en comun MAX | 6/7 (86%) | |
| Variable mas sensible | x3 (Delta H = -0.069) | |

---

## Enlaces

- **Video de Exposicion (10 min):** [YouTube](https://youtu.be/TU_LINK)
- **Informe PDF:** [Overleaf](https://www.overleaf.com/TU_LINK)
- **Repositorio:** [GitHub](https://github.com/liamcruzticona/ENTROPIA_PARTICION)
