"""Surtido y frentes de una categoría por análisis marginal.

Se agrega un frente a la vez, siempre el que más utilidad suma por metro.
Como la utilidad de cada SKU es cóncava en sus frentes, este orden da la
solución óptima de la relajación continua. Recorrerlo da además la curva
"utilidad de la categoría vs. metros de repisa" que usa el nivel macro.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from ..config import Modelo
from ..demanda import Sku, utilidad


@dataclass(frozen=True)
class Paso:
    sku: str
    frentes: int  # frentes del SKU después de este paso
    ancho: float
    valor: float  # utilidad que suma este paso


def pasos_marginales(skus: list[Sku], prof: float, ancho_max: float, modelo: Modelo) -> list[Paso]:
    heap = []
    por_id = {s.id: s for s in skus}
    for s in skus:
        v = utilidad(s, 1, 1.0, prof, modelo)
        if v > 0 and s.ancho <= ancho_max:
            heapq.heappush(heap, (-v / s.ancho, s.id, 1, v))
    pasos = []
    while heap:
        _, sid, f, v = heapq.heappop(heap)
        s = por_id[sid]
        pasos.append(Paso(sid, f, s.ancho, v))
        f_max = min(modelo.max_frentes, int(ancho_max // s.ancho))
        if f < f_max:
            dv = utilidad(s, f + 1, 1.0, prof, modelo) - utilidad(s, f, 1.0, prof, modelo)
            if dv > 1e-9:
                heapq.heappush(heap, (-dv / s.ancho, sid, f + 1, dv))
    return pasos


def seleccionar(pasos: list[Paso], capacidad: float) -> dict[str, int]:
    """Frentes por SKU al recorrer los pasos hasta llenar la capacidad (m)."""
    frentes: dict[str, int] = {}
    usado = 0.0
    for p in pasos:
        if frentes.get(p.sku, 0) == p.frentes - 1 and usado + p.ancho <= capacidad + 1e-9:
            frentes[p.sku] = p.frentes
            usado += p.ancho
    return frentes


def curva(pasos: list[Paso]) -> tuple[np.ndarray, np.ndarray]:
    """(metros acumulados, utilidad acumulada), empezando en (0, 0)."""
    w = np.concatenate([[0.0], np.cumsum([p.ancho for p in pasos])])
    v = np.concatenate([[0.0], np.cumsum([p.valor for p in pasos])])
    return w, v


def envolvente_concava(w: np.ndarray, v: np.ndarray, max_puntos: int = 30) -> list[tuple[float, float]]:
    """Envolvente cóncava superior de la curva, reducida a `max_puntos` vértices."""
    hull: list[tuple[float, float]] = []
    for p in zip(w, v):
        while len(hull) >= 2 and _cruz(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append((float(p[0]), float(p[1])))
    if len(hull) > max_puntos:
        idx = np.unique(np.linspace(0, len(hull) - 1, max_puntos).round().astype(int))
        hull = [hull[i] for i in idx]
    return hull


def _cruz(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
