"""Carga de los CSV que acompañan al plano: fichas de góndola, zonas y reglas de categoría."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon

from .estructura import Ficha
from .zonas import Zona


def cargar_fichas(ruta: Path) -> dict[str, Ficha]:
    df = pd.read_csv(ruta, dtype={"gondola_id": str, "niveles": str})
    fichas = {}
    for f in df.itertuples(index=False):
        niveles = tuple(sorted(float(v) for v in str(f.niveles).split(";") if v.strip()))
        fichas[f.gondola_id] = Ficha(
            tipo=f.tipo,
            alto=float(f.alto),
            niveles=niveles,
            ancho_tramo=float(f.ancho_tramo),
            prof_max_repisa=float(f.prof_max_repisa),
            espesor_repisa=float(f.espesor_repisa),
            caras_cortas=bool(int(f.caras_cortas)),
        )
    return fichas


def cargar_zonas(ruta: Path, poligonos: dict[str, Polygon], avisos: list[str]) -> list[Zona]:
    """Une los atributos de zonas.csv con los polígonos de la capa ZONA del plano."""
    df = pd.read_csv(ruta, dtype={"zona_id": str})
    zonas = []
    for f in df.itertuples(index=False):
        poly = poligonos.get(f.zona_id)
        if poly is None:
            avisos.append(f"Zona {f.zona_id} está en zonas.csv pero no en el plano; se ignora")
            continue
        if f.z_min >= f.z_max:
            avisos.append(f"Zona {f.zona_id}: z_min ({f.z_min}) ≥ z_max ({f.z_max}); se ignora")
            continue
        zonas.append(Zona(f.zona_id, str(f.tipo).upper(), poly, float(f.z_min), float(f.z_max), bool(int(f.exclusiva))))
    for zid in poligonos.keys() - set(df["zona_id"]):
        avisos.append(f"Zona {zid} está dibujada en el plano pero no tiene fila en zonas.csv; se ignora")
    return zonas


def cargar_categorias(dir_datos: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorias = pd.read_csv(dir_datos / "categorias.csv")
    reglas = pd.read_csv(dir_datos / "categoria_zona.csv")
    reglas["tipo_zona"] = reglas["tipo_zona"].str.upper()
    return categorias, reglas
