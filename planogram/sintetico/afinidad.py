"""Afinidad sintética entre categorías: qué conviene tener cerca y qué conviene separar.

Reemplaza a un paso futuro que la infiera (LLM, embeddings de nombres o
co-compra en boletas). Ese paso debe escribir el mismo archivo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ALIMENTOS = ("LAC", "FIA", "BEB", "CER", "CON", "HEL", "LIC", "SNA", "GAL", "ABA", "CNS", "DES")

# Pares que se compran juntos o que el cliente espera encontrar juntos.
AFINES = {
    ("CER", "SNA"): 1.0,
    ("CER", "BEB"): 0.8,
    ("CER", "LIC"): 0.6,
    ("BEB", "SNA"): 0.7,
    ("BEB", "LIC"): 0.4,
    ("SNA", "GAL"): 0.6,
    ("GAL", "DES"): 0.5,
    ("DES", "LAC"): 0.6,
    ("LAC", "FIA"): 0.7,
    ("ABA", "CNS"): 0.8,
    ("CON", "HEL"): 0.8,
    ("LIM", "CPE"): 0.7,
    ("LIM", "MAS"): 0.3,
}

# Químicos y no alimentos lejos de la comida.
SEPARAR = {"LIM": -0.8, "CPE": -0.4, "MAS": -0.2}


def generar_afinidad(dir_salida: Path) -> None:
    filas = [(a, b, p) for (a, b), p in AFINES.items()]
    filas += [(no_alimento, alimento, p) for no_alimento, p in SEPARAR.items() for alimento in ALIMENTOS]
    pd.DataFrame(
        [(a, b, p, "sintetico") for a, b, p in filas],
        columns=["categoria_a", "categoria_b", "peso", "fuente"],
    ).to_csv(dir_salida / "afinidad_categorias.csv", index=False)
