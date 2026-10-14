"""Qué categorías pueden ir en qué repisas según las zonas (restricción dura, se filtra antes de optimizar).

Regla:
- Repisa sin zona: acepta las categorías que no tienen zona obligatoria.
- Repisa en una zona de tipo T:
    - acepta las categorías PERMITIDA_EN T;
    - si la zona no es exclusiva, acepta además las categorías sin zona obligatoria.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

from .grafo import hijos, nodos


@dataclass(frozen=True)
class ReglaCategoria:
    permitidas: frozenset[str]  # tipos de zona donde puede ir
    obligatorias: frozenset[str]  # si no está vacío, solo puede ir en estos tipos


def reglas_categorias(G: nx.DiGraph) -> dict[str, ReglaCategoria]:
    reglas = {}
    for cat, _ in nodos(G, "categoria"):
        permitidas, obligatorias = set(), set()
        for tz in hijos(G, cat, "PERMITIDA_EN"):
            tipo = G.nodes[tz]["nombre"]
            permitidas.add(tipo)
            if G.edges[cat, tz]["obligatoria"]:
                obligatorias.add(tipo)
        reglas[cat] = ReglaCategoria(frozenset(permitidas), frozenset(obligatorias))
    return reglas


def zona_de(G: nx.DiGraph, repisa: str) -> str | None:
    zonas = hijos(G, repisa, "DENTRO_DE")
    return zonas[0] if zonas else None


def compatible(G: nx.DiGraph, regla: ReglaCategoria, repisa: str) -> bool:
    zona = zona_de(G, repisa)
    if zona is None:
        return not regla.obligatorias
    datos = G.nodes[zona]
    if datos["tipo_zona"] in regla.permitidas:
        return True
    return not regla.obligatorias and not datos["exclusiva"]


def repisas_compatibles(G: nx.DiGraph) -> dict[str, list[str]]:
    """categoría → repisas donde puede ir."""
    reglas = reglas_categorias(G)
    repisas = [r for r, _ in nodos(G, "repisa")]
    return {cat: [r for r in repisas if compatible(G, regla, r)] for cat, regla in reglas.items()}


def resumen_compatibilidad(G: nx.DiGraph) -> pd.DataFrame:
    filas = []
    for cat, repisas in repisas_compatibles(G).items():
        por_tipo: dict[str, float] = {}
        for r in repisas:
            zona = zona_de(G, r)
            tipo = G.nodes[zona]["tipo_zona"] if zona else "(sin zona)"
            por_tipo[tipo] = por_tipo.get(tipo, 0.0) + G.nodes[r]["ancho"]
        filas.append({
            "categoria": cat.removeprefix("CAT:"),
            "nombre": G.nodes[cat]["nombre"],
            "repisas": len(repisas),
            "metros_lineales": round(sum(por_tipo.values()), 1),
            "detalle": ", ".join(f"{t}: {m:.1f} m" for t, m in sorted(por_tipo.items())),
        })
    return pd.DataFrame(filas)
