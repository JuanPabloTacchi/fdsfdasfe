"""Etapa 3a con tráfico endógeno: dónde van las categorías ↔ por dónde camina la gente.

El layout de categorías determina el tráfico (los clientes van a buscar las
categorías de su lista) y el tráfico determina cuánto vende cada ubicación (las
compras por impulso dependen de cuánta gente pasa). Se busca un punto fijo:

    tráfico₀ = tráfico de referencia (aleatorio)
    iteración k:
        categorías_k = macro(tráfico_k)                    con la solución anterior como partida
        tráfico'_k   = simular(categorías_k)               el tráfico que ese layout genera
        valor_k      = valor de categorías_k con tráfico'_k  evaluación consistente
        tráfico_k+1  = mezcla(tráfico_k, tráfico'_k)       amortiguado
    hasta que cambie menos de `tolerancia_cambio` de las unidades.

Se devuelve la iteración de mayor valor, no necesariamente la última: el punto fijo
no garantiza que cada vuelta mejore. Ojo: esto encuentra un equilibrio, no el óptimo
global; el nivel macro toma el tráfico como dado y no "decide" mover una categoría
destino para arrastrar gente.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from ..config import Modelo
from ..demanda import ParamsCategoria
from ..scoring import Simulador, Trafico, aplicar_trafico, mezclar
from .macro import CurvaCategoria, resolver_macro
from .unidades import Unidad, actualizar_exposicion


def valor_estimado(unidades: list[Unidad], asignacion: dict[str, str], curvas: dict[str, CurvaCategoria],
                   frac_impulso: dict[str, float]) -> float:
    """Utilidad semanal estimada de una asignación: curva de la categoría × multiplicador medio de su espacio."""
    total = 0.0
    for c, curva in curvas.items():
        us = [u for u in unidades if asignacion.get(u.id) == c]
        metros = sum(u.capacidad for u in us)
        if not metros:
            continue
        w, v = zip(*curva.puntos)
        mult = sum(u.capacidad * u.multiplicador(frac_impulso.get(c, 0.0)) for u in us) / metros
        total += float(np.interp(metros, w, v)) * mult
    return total


def optimizar_categorias(
    G: nx.DiGraph,
    sim: Simulador,
    unidades: list[Unidad],
    curvas: dict[str, CurvaCategoria],
    pares: list[tuple[int, int]],
    vecinos: list[tuple[int, int, str]],
    afinidad: dict[tuple[str, str], float],
    params: dict[str, ParamsCategoria],
    trafico_inicial: Trafico,
    referencia: float,
    modelo: Modelo,
) -> tuple[dict[str, str], Trafico, list[dict], dict]:
    """Devuelve (asignación, su tráfico, historial de iteraciones, estadísticas del solver de la elegida).

    Al terminar, el grafo y las unidades quedan con el tráfico de la asignación elegida.
    """
    frac = {c: p.frac_impulso for c, p in params.items()}
    trafico = trafico_inicial
    previa: dict[str, str] | None = None
    historial: list[dict] = []
    candidatas = []

    for k in range(modelo.iteraciones_trafico + 1):
        t0 = time.perf_counter()
        aplicar_trafico(G, trafico, modelo, referencia)
        actualizar_exposicion(G, unidades)
        asignacion, stats = resolver_macro(unidades, curvas, pares, modelo, vecinos, afinidad, frac, previa)

        # Tráfico que genera este layout y valor del layout con ese tráfico (evaluación consistente).
        if modelo.iteraciones_trafico == 0:
            propio, st_sim = trafico, {"recorrido_medio_m": None}
        else:
            propio, st_sim = sim.por_listas(unidades, asignacion, params)
        aplicar_trafico(G, propio, modelo, referencia)
        actualizar_exposicion(G, unidades)
        valor = valor_estimado(unidades, asignacion, curvas, frac)

        cambio = 1.0 if previa is None else (
            sum(asignacion.get(u.id) != previa.get(u.id) for u in unidades) / len(unidades))
        historial.append({
            "iteracion": k,
            "valor_estimado": round(valor, 0),
            "unidades_cambiadas": round(cambio, 3) if previa is not None else None,
            "recorrido_medio_m": st_sim["recorrido_medio_m"],
            "brecha_macro": round(stats["brecha"], 4),
            "segundos": round(time.perf_counter() - t0, 1),
        })
        candidatas.append((valor, k, asignacion, propio, stats))
        if previa is not None and cambio < modelo.tolerancia_cambio:
            break
        previa = asignacion
        trafico = mezclar(trafico, propio, modelo.mezcla_trafico)

    valor, k, asignacion, propio, stats = max(candidatas, key=lambda t: t[0])
    for h in historial:
        h["elegida"] = h["iteracion"] == k
    aplicar_trafico(G, propio, modelo, referencia)
    actualizar_exposicion(G, unidades)
    return asignacion, propio, historial, stats
