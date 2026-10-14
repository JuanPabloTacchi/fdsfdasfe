"""Zonas: volúmenes (polígono en planta × rango de altura) que restringen qué va en las repisas."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from shapely.geometry import Polygon

from ..config import Config
from .estructura import Repisa


@dataclass(frozen=True)
class Zona:
    id: str
    tipo: str
    poligono: Polygon
    z_min: float
    z_max: float
    # Exclusiva: solo acepta categorías permitidas en su tipo.
    # No exclusiva: acepta además cualquier categoría que no tenga zona obligatoria.
    exclusiva: bool


def fraccion_dentro(repisa: Repisa, zona: Zona) -> float:
    """Fracción del volumen libre de la repisa (huella × [z, z + hueco]) que cae dentro de la zona."""
    if not repisa.huella.intersects(zona.poligono):
        return 0.0
    f_planta = repisa.huella.intersection(zona.poligono).area / repisa.huella.area
    z0, z1 = repisa.z, repisa.z + repisa.hueco
    f_altura = max(0.0, min(z1, zona.z_max) - max(z0, zona.z_min)) / (z1 - z0)
    return f_planta * f_altura


def asignar_zonas(
    repisas: Iterable[Repisa],
    zonas: list[Zona],
    cfg: Config,
    avisos: list[str],
) -> dict[str, tuple[str, float]]:
    """Devuelve repisa_id → (zona_id, fracción) para cada repisa que cae dentro de una zona."""
    asignacion: dict[str, tuple[str, float]] = {}
    for r in repisas:
        fracs = [(z.id, fraccion_dentro(r, z)) for z in zonas]
        fracs = [(zid, f) for zid, f in fracs if f > cfg.tolerancia_parcial]
        if not fracs:
            continue
        for zid, f in fracs:
            if f < 1 - cfg.tolerancia_parcial:
                destino = "queda dentro" if f >= cfg.umbral_contencion else "queda fuera"
                avisos.append(f"{r.id}: solo el {f:.0%} de su volumen cae en {zid} ({destino}); revisar el dibujo de la zona")
        dentro = [(zid, f) for zid, f in fracs if f >= cfg.umbral_contencion]
        if len(dentro) > 1:
            avisos.append(f"{r.id}: cae en varias zonas {[z for z, _ in dentro]}; se asigna a la de mayor fracción")
        if dentro:
            asignacion[r.id] = max(dentro, key=lambda t: t[1])
    return asignacion
