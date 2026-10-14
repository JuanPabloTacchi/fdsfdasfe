"""Layout de referencia con reglas simples, para comparar contra el optimizado.

Imita la práctica habitual sin optimización:
- Macro: cada categoría obligada a una zona va a esa zona; el resto, al espacio
  libre. Dentro de cada grupo, las categorías se reparten las unidades en orden
  de lectura (góndola, cara, tramo), con espacio proporcional a su utilidad.
- Micro: SKUs de mayor a menor venta con sus frentes de referencia, colocados
  en la primera repisa donde quepan (de abajo hacia arriba, tramo por tramo).
"""

from __future__ import annotations

import networkx as nx

from ..demanda import Sku, cabe, multiplicador, utilidad
from ..config import Modelo
from ..ingesta.compatibilidad import reglas_categorias
from .micro import Ubicacion, _ordenar_repisa
from .unidades import Unidad


def macro_base(G: nx.DiGraph, unidades: list[Unidad], skus: dict[str, Sku]) -> dict[str, str]:
    reglas = {c.removeprefix("CAT:"): r for c, r in reglas_categorias(G).items()}
    valor_cat: dict[str, float] = {}
    for s in skus.values():
        valor_cat[s.categoria] = valor_cat.get(s.categoria, 0.0) + s.margen * s.demanda

    grupos: dict[str | None, list[Unidad]] = {}
    for u in unidades:
        tipo = G.nodes[u.zona]["tipo_zona"] if u.zona else None
        grupos.setdefault(tipo, []).append(u)

    asignacion = {}
    for tipo, us in grupos.items():
        if tipo is None:
            cats = [c for c, r in reglas.items() if not r.obligatorias]
        else:
            cats = [c for c, r in reglas.items() if tipo in r.obligatorias] or \
                   [c for c, r in reglas.items() if tipo in r.permitidas]
        cats = sorted(c for c in cats if valor_cat.get(c, 0) > 0)
        if not cats:
            continue
        us = sorted(us, key=lambda u: (u.cara, u.orden))
        total = sum(u.capacidad for u in us)
        total_valor = sum(valor_cat[c] for c in cats)
        k, acumulado, meta = 0, 0.0, total * valor_cat[cats[0]] / total_valor
        for u in us:
            while acumulado >= meta - 1e-9 and k < len(cats) - 1:
                k += 1
                meta += total * valor_cat[cats[k]] / total_valor
            asignacion[u.id] = cats[k]
            acumulado += u.capacidad
    return asignacion


def micro_base(G: nx.DiGraph, repisas: list[str], skus: list[Sku], modelo: Modelo,
               frac_impulso: float = 0.0) -> list[Ubicacion]:
    R = {r: G.nodes[r] for r in repisas}
    libre = {r: d["ancho"] for r, d in R.items()}
    objetivo = sum(libre.values())
    skus = [s for s in skus if any(cabe(s, d) for d in R.values())]

    # Frentes proporcionales a la venta, escalados para ocupar todo el espacio de la categoría.
    def frentes(alfa: float) -> dict[str, int]:
        return {s.id: min(modelo.max_frentes, max(1, round(alfa * s.demanda))) for s in skus}

    lo, hi = 0.0, 100.0
    for _ in range(40):
        alfa = (lo + hi) / 2
        if sum(s.ancho * f for s, f in zip(skus, frentes(alfa).values())) > objetivo:
            hi = alfa
        else:
            lo = alfa
    f_obj = frentes(lo)

    ubicaciones = []
    for s in sorted(skus, key=lambda s: -s.demanda):
        for f in range(f_obj[s.id], 0, -1):
            r = next((r for r in repisas if cabe(s, R[r], f) and libre[r] >= s.ancho * f - 1e-9), None)
            if r:
                ubicaciones.append(Ubicacion(r, s.id, f))
                libre[r] -= s.ancho * f
                break
    por_id = {s.id: s for s in skus}
    # Espacio sobrante: un frente más al SKU con más venta por frente, hasta llenar la repisa.
    for r in repisas:
        en_repisa = [ub for ub in ubicaciones if ub.repisa == r]
        while True:
            opciones = [ub for ub in en_repisa if ub.frentes < modelo.max_frentes
                        and por_id[ub.sku].ancho <= libre[r] + 1e-9]
            if not opciones:
                break
            ub = max(opciones, key=lambda ub: por_id[ub.sku].demanda / ub.frentes)
            ub.frentes += 1
            libre[r] -= por_id[ub.sku].ancho
    resultado = []
    for r in repisas:
        resultado += _ordenar_repisa([ub for ub in ubicaciones if ub.repisa == r], por_id)
    for ub in resultado:
        r = R[ub.repisa]
        ub.utilidad = utilidad(por_id[ub.sku], ub.frentes, multiplicador(r, frac_impulso, modelo), r["profundidad"], modelo)
    return resultado
