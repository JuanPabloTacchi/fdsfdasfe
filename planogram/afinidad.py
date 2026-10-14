"""Afinidad entre categorías: peso en [-1, 1] por par, simétrico.

Formato de afinidad_categorias.csv:

    categoria_a, categoria_b, peso, fuente

- peso > 0: conviene que estén cerca (se compran juntas, el cliente las busca juntas).
- peso < 0: conviene separarlas (químicos y alimentos, por ejemplo).
- fuente: de dónde salió el peso (sintetico, llm, boletas, manual...). Solo informativo.

Los pares que no aparecen pesan 0. Si un par aparece dos veces, vale el último.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

Afinidad = dict[tuple[str, str], float]


def peso_vecindad(afinidad: Afinidad, c: str, d: str, tipo: str) -> float:
    """Peso que cuenta para dos categorías vecinas, según cómo son vecinas.

    - "frente" (enfrentadas en un pasillo): cuenta el peso completo.
    - "lado" (contiguas en la misma cara): solo cuentan los pesos negativos. Si se premiara
      la afinidad positiva, convendría intercalar categorías tramo por tramo (más fronteras),
      y el cliente encontraría cada categoría repartida en franjas.
    """
    if c == d:
        return 0.0
    peso = afinidad.get((c, d), 0.0)
    return peso if tipo == "frente" or peso < 0 else 0.0


def cargar_afinidad(ruta: Path, categorias: set[str], avisos: list[str]) -> Afinidad:
    if not ruta.exists():
        avisos.append(f"No existe {ruta.name}: no se considera afinidad entre categorías")
        return {}
    afinidad: Afinidad = {}
    for f in pd.read_csv(ruta).itertuples(index=False):
        a, b, peso = str(f.categoria_a), str(f.categoria_b), float(f.peso)
        if a not in categorias or b not in categorias:
            avisos.append(f"Afinidad {a}-{b}: categoría desconocida, se ignora")
            continue
        if a == b:
            continue  # la contigüidad de una misma categoría se maneja aparte
        if not -1 <= peso <= 1:
            avisos.append(f"Afinidad {a}-{b}: peso {peso} fuera de [-1, 1], se recorta")
            peso = max(-1.0, min(1.0, peso))
        afinidad[a, b] = afinidad[b, a] = peso
    return afinidad
