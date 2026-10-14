"""Navmesh en grilla: divide el área caminable en celdas cuadradas conectadas a sus vecinas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import shapely
from shapely.geometry import Polygon


@dataclass
class Celda:
    id: str
    i: int
    j: int
    centro: tuple[float, float]
    area: float
    holgura: float  # distancia del centro al obstáculo más cercano
    poligono: Polygon


@dataclass
class Navmesh:
    celdas: dict[str, Celda]
    vecinos: list[tuple[str, str, float]]  # (celda_a, celda_b, distancia), cada par una vez

    @cached_property
    def ids(self) -> list[str]:
        return list(self.celdas)

    @cached_property
    def puntos(self) -> np.ndarray:
        """Centros de las celdas como puntos de shapely, en el mismo orden que `ids`."""
        return shapely.points([c.centro for c in self.celdas.values()])


def construir_navmesh(caminable: Polygon, tam: float, frac_min: float) -> Navmesh:
    minx, miny, maxx, maxy = caminable.bounds
    ni, nj = math.ceil((maxx - minx) / tam), math.ceil((maxy - miny) / tam)
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    x0, y0 = minx + ii * tam, miny + jj * tam
    cajas = shapely.box(x0, y0, x0 + tam, y0 + tam)

    shapely.prepare(caminable)
    candidatas = shapely.intersects(caminable, cajas)
    inter = np.empty(len(cajas), dtype=object)
    inter[candidatas] = shapely.intersection(cajas[candidatas], caminable)
    areas = np.zeros(len(cajas))
    areas[candidatas] = shapely.area(inter[candidatas])
    quedan = np.flatnonzero(areas >= frac_min * tam * tam)

    centros = shapely.centroid(inter[quedan])
    holguras = shapely.distance(caminable.boundary, centros)

    celdas: dict[str, Celda] = {}
    por_ij: dict[tuple[int, int], str] = {}
    for k, idx in enumerate(quedan):
        i, j = int(ii[idx]), int(jj[idx])
        cid = f"P:{i}_{j}"
        c = centros[k]
        celdas[cid] = Celda(cid, i, j, (c.x, c.y), float(areas[idx]), float(holguras[k]), inter[idx])
        por_ij[(i, j)] = cid

    vecinos = []
    for (i, j), a in por_ij.items():
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            b = por_ij.get((i + di, j + dj))
            if b is None:
                continue
            # En diagonal solo si las dos celdas ortogonales existen: no se cortan esquinas de góndolas.
            if di and dj and ((i + di, j) not in por_ij or (i, j + dj) not in por_ij):
                continue
            vecinos.append((a, b, math.dist(celdas[a].centro, celdas[b].centro)))
    return Navmesh(celdas, vecinos)
