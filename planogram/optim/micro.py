"""Etapa 3b: planograma de una categoría dentro del espacio que le asignó el nivel macro.

1. Surtido y frentes: análisis marginal sobre `holgura_micro` × los metros asignados.
2. Ubicación (CP-SAT, asignación generalizada): cada SKU elegido, con sus frentes,
   va a lo más a una repisa donde quepa en alto, fondo y ancho, maximizando
   Σ utilidad(SKU, frentes, multiplicador de la repisa, fondo de la repisa).
3. Relleno: los SKUs que no entraron se prueban con menos frentes y el espacio
   que sobra en cada repisa se usa para el frente adicional que más utilidad sume.
4. Posición horizontal: en cada repisa, bloques por marca (la de más utilidad
   primero, a la izquierda) y dentro de cada marca de mayor a menor utilidad.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import networkx as nx
from ortools.sat.python import cp_model

from ..config import Modelo
from ..demanda import Sku, cabe, multiplicador, utilidad
from .surtido import pasos_marginales, seleccionar

MM = 1000
CENT = 100


@dataclass
class Ubicacion:
    repisa: str
    sku: str
    frentes: int
    x0: float = 0.0
    utilidad: float = 0.0


def resolver_micro(
    G: nx.DiGraph,
    repisas: list[str],
    skus: list[Sku],
    modelo: Modelo,
    frac_impulso: float = 0.0,
) -> tuple[list[Ubicacion], dict]:
    R = {r: G.nodes[r] for r in repisas}
    mult = {r: multiplicador(d, frac_impulso, modelo) for r, d in R.items()}
    skus = [s for s in skus if any(cabe(s, d) for d in R.values())]
    if not skus or not R:
        return [], {"estado": "SIN_SKUS", "skus_candidatos": len(skus)}

    def u(s: Sku, f: int, r: str) -> float:
        return utilidad(s, f, mult[r], R[r]["profundidad"], modelo)

    capacidad = sum(d["ancho"] for d in R.values())
    prof = statistics.median(d["profundidad"] for d in R.values())
    ancho_max = max(d["ancho"] for d in R.values())
    frentes = seleccionar(pasos_marginales(skus, prof, ancho_max, modelo), modelo.holgura_micro * capacidad)
    por_id = {s.id: s for s in skus}

    # Candidatas de cada SKU elegido: repisas donde cabe con sus frentes, las de más utilidad primero.
    candidatas: dict[str, list[tuple[str, float]]] = {}
    for sid, f in frentes.items():
        s = por_id[sid]
        while f >= 1:
            opciones = sorted(((r, u(s, f, r)) for r in R if cabe(s, R[r], f)), key=lambda t: -t[1])
            if opciones:
                frentes[sid] = f
                # Sin poda por defecto: si cada SKU ve solo sus mejores repisas, todos compiten por
                # las mismas (las de altura de ojos) y el resto queda vacío.
                candidatas[sid] = [(r, v) for r, v in opciones[: modelo.candidatas_por_sku] if v > 0]
                break
            f -= 1

    m = cp_model.CpModel()
    y = {(sid, r): m.new_bool_var(f"y_{sid}_{r}") for sid, ops in candidatas.items() for r, _ in ops}
    for sid, ops in candidatas.items():
        m.add_at_most_one(y[sid, r] for r, _ in ops)
    por_repisa: dict[str, list[tuple[str, int]]] = {}
    for sid, ops in candidatas.items():
        for r, _ in ops:
            # Redondeo conservador: anchos hacia arriba, capacidad hacia abajo.
            por_repisa.setdefault(r, []).append((sid, math.ceil(por_id[sid].ancho * frentes[sid] * MM - 1e-9)))
    for r, items in por_repisa.items():
        m.add(sum(y[sid, r] * w for sid, w in items) <= math.floor(R[r]["ancho"] * MM + 1e-9))
    m.maximize(sum(y[sid, r] * round(v * CENT) for sid, ops in candidatas.items() for r, v in ops))

    pista = _pista(candidatas, frentes, por_id, R)
    for (sid, r), var in pista.items():
        m.add_hint(y[sid, r], var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = modelo.tiempo_micro
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = modelo.semilla
    estado = solver.solve(m)
    if estado in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        elegidas = [k for k, var in y.items() if solver.value(var)]
        nombre_estado = solver.status_name(estado)
    else:
        # Sin solución a tiempo (pasa con modelos grandes y poco tiempo): se usa la codiciosa.
        elegidas = [k for k, v in pista.items() if v]
        nombre_estado = f"CODICIOSA ({solver.status_name(estado)})"

    ubicaciones = {sid: Ubicacion(r, sid, frentes[sid]) for sid, r in elegidas}
    sin_ubicar_cp = len(candidatas) - len(ubicaciones)
    _rellenar(ubicaciones, skus, R, mult, u, modelo)

    resultado = []
    for r in R:
        resultado += _ordenar_repisa([ub for ub in ubicaciones.values() if ub.repisa == r], por_id)
    for ub in resultado:
        ub.utilidad = u(por_id[ub.sku], ub.frentes, ub.repisa)

    stats = {
        "estado": nombre_estado,
        "skus_candidatos": len(skus),
        "skus_elegidos": len(frentes),
        "skus_sin_ubicar_cp": sin_ubicar_cp,
        "skus_ubicados": len(resultado),
        "variables": len(y),
        "segundos": solver.wall_time,
    }
    return resultado, stats


def _pista(candidatas, frentes, por_id, R) -> dict[tuple[str, str], int]:
    """Solución codiciosa para arrancar el solver: cada SKU en su mejor repisa con espacio libre."""
    libre = {r: d["ancho"] for r, d in R.items()}
    pista = {}
    orden = sorted(candidatas, key=lambda sid: -(candidatas[sid][0][1] if candidatas[sid] else 0))
    for sid in orden:
        w = por_id[sid].ancho * frentes[sid]
        elegida = next((r for r, _ in candidatas[sid] if libre[r] >= w - 1e-9), None)
        for r, _ in candidatas[sid]:
            pista[sid, r] = int(r == elegida)
        if elegida:
            libre[elegida] -= w
    return pista


def _rellenar(ubicaciones: dict[str, Ubicacion], skus: list[Sku], R: dict, mult: dict, u, modelo: Modelo) -> None:
    por_id = {s.id: s for s in skus}
    libre = {r: d["ancho"] for r, d in R.items()}
    for ub in ubicaciones.values():
        libre[ub.repisa] -= por_id[ub.sku].ancho * ub.frentes

    # Repisas más valiosas primero; en cada una se suma el frente (o SKU nuevo) con más utilidad por metro.
    for r in sorted(R, key=lambda r: -mult[r]):
        while True:
            mejor, mejor_ratio = None, 0.0
            for s in skus:
                ub = ubicaciones.get(s.id)
                if ub is not None and ub.repisa != r:
                    continue
                f = ub.frentes if ub else 0
                if f >= modelo.max_frentes or s.ancho > libre[r] + 1e-9 or not cabe(s, R[r], f + 1):
                    continue
                ganancia = u(s, f + 1, r) - u(s, f, r)
                if ganancia / s.ancho > mejor_ratio:
                    mejor, mejor_ratio = s, ganancia / s.ancho
            if mejor is None:
                break
            if mejor.id in ubicaciones:
                ubicaciones[mejor.id].frentes += 1
            else:
                ubicaciones[mejor.id] = Ubicacion(r, mejor.id, 1)
            libre[r] -= mejor.ancho


def _ordenar_repisa(items: list[Ubicacion], por_id: dict[str, Sku]) -> list[Ubicacion]:
    valor = {ub.sku: por_id[ub.sku].margen * por_id[ub.sku].demanda for ub in items}
    por_marca: dict[str, float] = {}
    for ub in items:
        marca = por_id[ub.sku].marca
        por_marca[marca] = por_marca.get(marca, 0.0) + valor[ub.sku]
    items.sort(key=lambda ub: (-por_marca[por_id[ub.sku].marca], por_id[ub.sku].marca, -valor[ub.sku]))
    x = 0.0
    for ub in items:
        ub.x0 = x
        x += por_id[ub.sku].ancho * ub.frentes
    return items
