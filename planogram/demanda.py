"""Modelo de demanda y utilidad. Lo usan tanto la optimización como la evaluación.

Utilidad semanal de un SKU con f frentes en una repisa:

    venta     = demanda_base × multiplicador_repisa × (f / f_ref)^beta
    capacidad = f × unidades_al_fondo × reposiciones_por_semana
    utilidad  = margen_unitario × min(venta, capacidad)

- demanda_base: promedio semanal histórico de la tienda. Se asume que se observó
  con f_ref frentes en una ubicación promedio (multiplicador 1).
- f_ref: frentes que hacen falta para no quedarse sin stock entre reposiciones.
- multiplicador_repisa: depende de la repisa y de la categoría (ver `multiplicador`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Modelo

PROF_TIPICA = 0.5  # m, fondo de repisa con que se calcula f_ref

# Por rol de la categoría: (probabilidad de estar en la lista de compras, fracción de la venta que es impulso).
# Valores supuestos; con boletas reales se estiman (qué fracción de las boletas incluye la categoría).
PARAMS_POR_ROL = {
    "destino": (0.55, 0.10),
    "rutina": (0.30, 0.25),
    "conveniencia": (0.15, 0.40),
    "impulso": (0.08, 0.65),
}


@dataclass(frozen=True)
class ParamsCategoria:
    prob_lista: float  # probabilidad de que un cliente venga a buscarla (la tenga en su lista)
    frac_impulso: float  # fracción de su venta que depende de que el cliente pase frente a ella


def params_categorias(categorias: pd.DataFrame, avisos: list[str]) -> dict[str, ParamsCategoria]:
    """Lee prob_lista y frac_impulso de categorias.csv; si faltan, usa los valores por rol."""
    tiene = {"prob_lista", "frac_impulso"} <= set(categorias.columns)
    if not tiene:
        avisos.append("categorias.csv no trae prob_lista / frac_impulso: se usan valores por rol")
    params = {}
    for f in categorias.itertuples(index=False):
        p, a = PARAMS_POR_ROL.get(f.rol, (0.2, 0.3))
        if tiene:
            p, a = float(f.prob_lista), float(f.frac_impulso)
        params[f.categoria_id] = ParamsCategoria(p, a)
    return params


def multiplicador(repisa: dict, frac_impulso: float, modelo: Modelo) -> float:
    """Cuánto vende una repisa respecto de una ubicación promedio, para una categoría.

    La parte planificada de la venta no depende del tráfico (el cliente viene a buscarla);
    la de impulso es proporcional a la exposición (pasadas frente a la repisa, relativas al
    tráfico de referencia). La visibilidad (altura, cabecera) afecta a ambas.
    """
    exposicion = repisa.get("exposicion", 1.0)
    return ((1 - frac_impulso) + frac_impulso * exposicion) * repisa.get("visibilidad_rel", 1.0) ** modelo.gamma_posicion


@dataclass(frozen=True)
class Sku:
    id: str
    categoria: str
    marca: str
    ancho: float
    alto: float
    prof: float
    margen: float  # $ por unidad
    demanda: float  # unidades por semana
    frentes_ref: int


def unidades_al_fondo(prof_repisa: float, prof_sku: float) -> int:
    # Tolerancia: en punto flotante 0.5 // 0.1 da 4.
    return int(prof_repisa / prof_sku + 1e-9)


def cargar_skus(dir_datos: Path, tienda: str, modelo: Modelo) -> dict[str, Sku]:
    skus = pd.read_csv(dir_datos / "skus.csv")
    ventas = pd.read_csv(dir_datos / "ventas.csv")
    ventas = ventas[ventas["tienda_id"] == tienda]
    semanas = ventas["semana"].nunique()
    demanda = ventas.groupby("sku_id")["unidades"].sum() / max(semanas, 1)

    resultado = {}
    for f in skus.itertuples(index=False):
        d = float(demanda.get(f.sku_id, 0.0))
        fondo = max(1, unidades_al_fondo(PROF_TIPICA, f.prof))
        f_ref = min(modelo.max_frentes, max(1, math.ceil(d / (modelo.reposiciones_semana * fondo))))
        resultado[f.sku_id] = Sku(f.sku_id, f.categoria_id, f.marca, f.ancho, f.alto, f.prof,
                                  f.precio - f.costo, d, f_ref)
    return resultado


def cabe(sku: Sku, repisa: dict, frentes: int = 1) -> bool:
    return (sku.alto <= repisa["hueco"] and sku.prof <= repisa["profundidad"]
            and sku.ancho * frentes <= repisa["ancho"] + 1e-9)


def venta(sku: Sku, frentes: int, multiplicador: float, prof_repisa: float, modelo: Modelo) -> float:
    """Unidades por semana: la demanda, limitada por lo que cabe en la repisa entre reposiciones."""
    if frentes <= 0:
        return 0.0
    fondo = unidades_al_fondo(prof_repisa, sku.prof)
    if fondo == 0 or sku.demanda <= 0:
        return 0.0
    demanda = sku.demanda * multiplicador * (frentes / sku.frentes_ref) ** modelo.beta_espacio
    capacidad = frentes * fondo * modelo.reposiciones_semana
    return min(demanda, capacidad)


def utilidad(sku: Sku, frentes: int, multiplicador: float, prof_repisa: float, modelo: Modelo) -> float:
    return sku.margen * venta(sku, frentes, multiplicador, prof_repisa, modelo)
