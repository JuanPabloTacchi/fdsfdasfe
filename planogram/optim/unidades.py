"""Unidades de espacio: lo que el nivel macro asigna a una categoría.

Una unidad es un tramo completo o, si una zona lo corta, la parte del tramo
dentro de cada zona (por ejemplo, las dos repisas superiores con llave y las
tres de abajo libres son dos unidades distintas del mismo tramo).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..ingesta.compatibilidad import compatible, reglas_categorias, zona_de
from ..ingesta.grafo import hijos, nodos


@dataclass
class Unidad:
    id: str
    cara: str
    tramo: str
    orden: int
    zona: str | None
    repisas: list[str]
    capacidad: float  # metros de repisa (suma de anchos)
    exposicion: float  # tráfico relativo a la referencia, promedio ponderado por ancho
    visibilidad: float  # visibilidad_rel^gamma, promedio ponderado por ancho
    categorias: list[str]  # compatibles, sin el prefijo CAT:

    def multiplicador(self, frac_impulso: float) -> float:
        """Multiplicador de demanda de la unidad para una categoría (ver demanda.multiplicador)."""
        return ((1 - frac_impulso) + frac_impulso * self.exposicion) * self.visibilidad


def construir_unidades(G: nx.DiGraph, gamma: float) -> list[Unidad]:
    reglas = reglas_categorias(G)
    unidades = []
    for c, _ in nodos(G, "cara"):
        for t in hijos(G, c, "TIENE_TRAMO"):
            grupos: dict[str | None, list[str]] = {}
            for r in hijos(G, t, "TIENE_REPISA"):
                grupos.setdefault(zona_de(G, r), []).append(r)
            for zona, repisas in grupos.items():
                anchos = np.array([G.nodes[r]["ancho"] for r in repisas])
                expo = np.array([G.nodes[r].get("exposicion", 1.0) for r in repisas])
                vis = np.array([G.nodes[r].get("visibilidad_rel", 1.0) ** gamma for r in repisas])
                cats = [cat.removeprefix("CAT:") for cat, regla in reglas.items() if compatible(G, regla, repisas[0])]
                unidades.append(Unidad(
                    id=f"{t.removeprefix('T:')}|{zona.removeprefix('Z:') if zona else '-'}",
                    cara=c, tramo=t, orden=G.nodes[t]["orden"], zona=zona, repisas=repisas,
                    capacidad=float(anchos.sum()),
                    exposicion=float((anchos * expo).sum() / anchos.sum()),
                    visibilidad=float((anchos * vis).sum() / anchos.sum()),
                    categorias=cats,
                ))
    return unidades


def adyacencias(unidades: list[Unidad]) -> list[tuple[int, int]]:
    """Pares de unidades vecinas: tramos consecutivos de la misma cara y con la misma zona."""
    indice = {(u.cara, u.orden, u.zona): k for k, u in enumerate(unidades)}
    pares = []
    for k, u in enumerate(unidades):
        vecina = indice.get((u.cara, u.orden + 1, u.zona))
        if vecina is not None:
            pares.append((k, vecina))
    return pares


def vecindades(G: nx.DiGraph, unidades: list[Unidad], ancho_max_pasillo: float) -> list[tuple[int, int, str]]:
    """Pares de unidades que un cliente ve juntas, para la afinidad entre categorías.

    - "lado": misma cara, mismo tramo o tramos consecutivos (cualquier zona).
    - "frente": tramos de caras enfrentadas en un mismo pasillo, a menos de `ancho_max_pasillo`
      y sin una góndola de por medio.
    """
    pares: list[tuple[int, int, str]] = []
    por_cara: dict[str, list[int]] = {}
    por_tramo: dict[str, list[int]] = {}
    for k, u in enumerate(unidades):
        por_cara.setdefault(u.cara, []).append(k)
        por_tramo.setdefault(u.tramo, []).append(k)
    for ks in por_cara.values():
        for i in ks:
            for j in ks:
                if i < j and abs(unidades[i].orden - unidades[j].orden) <= 1:
                    pares.append((i, j, "lado"))

    gondolas = unary_union([Polygon(d["poligono"]) for _, d in nodos(G, "gondola")])
    shapely.prepare(gondolas)
    tramos = []
    for t in por_tramo:
        d, cara = G.nodes[t], G.nodes[unidades[por_tramo[t][0]].cara]
        p0, p1 = np.array(d["p0"]), np.array(d["p1"])
        u = (p1 - p0) / d["ancho"]
        tramos.append((t, (p0 + p1) / 2, np.array(cara["normal"]), u, d["ancho"], unidades[por_tramo[t][0]].cara))
    for i, (t1, m1, n1, u1, w1, c1) in enumerate(tramos):
        for t2, m2, n2, _, w2, c2 in tramos[i + 1:]:
            if c1 == c2 or float(n1 @ n2) > -0.7:
                continue
            delta = m2 - m1
            dist = float(np.linalg.norm(delta))
            if dist > ancho_max_pasillo or float(delta @ n1) < 0.5 * dist or float(-delta @ n2) < 0.5 * dist:
                continue
            if abs(float(delta @ u1)) >= (w1 + w2) / 2 - 1e-6:
                continue
            visual = shapely.LineString([m1 + n1 * 0.05, m2 + n2 * 0.05])
            if gondolas.intersects(visual):
                continue
            pares += [(a, b, "frente") for a in por_tramo[t1] for b in por_tramo[t2]]
    return pares


def actualizar_exposicion(G: nx.DiGraph, unidades: list[Unidad]) -> None:
    """Refresca la exposición de cada unidad después de cambiar el tráfico (la visibilidad no cambia)."""
    for u in unidades:
        anchos = np.array([G.nodes[r]["ancho"] for r in u.repisas])
        expo = np.array([G.nodes[r].get("exposicion", 1.0) for r in u.repisas])
        u.exposicion = float((anchos * expo).sum() / anchos.sum())
