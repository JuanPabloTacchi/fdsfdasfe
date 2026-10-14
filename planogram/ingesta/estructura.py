"""Descompone cada góndola en caras, tramos y repisas.

Convenciones:
- Cada góndola se aproxima por su rectángulo mínimo y se recorre en sentido
  antihorario, así la normal (dy, -dx) de cada lado apunta hacia afuera.
- Una cara va de p0 a p1, que es de izquierda a derecha para quien la mira de
  frente. x0 de tramos y repisas se mide en ese sentido desde p0.
- La cara se nombra por la orientación de su normal: E, N, O, S (+y = N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from ..config import Config

Punto = tuple[float, float]
CARDINALES = ("E", "N", "O", "S")


@dataclass(frozen=True)
class Ficha:
    """Datos de una góndola que no vienen en el plano."""

    tipo: str
    alto: float
    niveles: tuple[float, ...]  # altura de cada repisa desde el piso, de abajo hacia arriba
    ancho_tramo: float
    prof_max_repisa: float
    espesor_repisa: float
    caras_cortas: bool  # si los extremos de la góndola también exhiben producto


FICHA_POR_DEFECTO = Ficha("lineal", 2.0, (0.15, 0.55, 0.95, 1.35, 1.75), 1.0, 0.6, 0.02, False)


@dataclass
class Repisa:
    id: str
    nivel: int  # 1 = la más baja
    z: float
    hueco: float  # alto libre hasta la repisa de arriba (o el techo de la góndola)
    x0: float
    ancho: float
    profundidad: float
    huella: Polygon


@dataclass
class Tramo:
    id: str
    orden: int  # 1 = el de más a la izquierda mirando la cara
    x0: float
    ancho: float
    p0: Punto
    p1: Punto
    repisas: list[Repisa] = field(default_factory=list)


@dataclass
class Cara:
    id: str
    orientacion: str
    lado: str  # "largo" | "corto"
    p0: Punto
    p1: Punto
    normal: Punto
    longitud: float
    profundidad: float
    frac_frente_caminable: float
    tramos: list[Tramo] = field(default_factory=list)


def validar_ficha(gid: str, ficha: Ficha) -> list[str]:
    avisos = []
    if not ficha.niveles:
        avisos.append(f"{gid}: la ficha no tiene niveles de repisa")
    elif ficha.niveles[-1] >= ficha.alto:
        avisos.append(f"{gid}: la repisa más alta ({ficha.niveles[-1]} m) supera el alto de la góndola ({ficha.alto} m)")
    for z, z_sig in zip(ficha.niveles, ficha.niveles[1:]):
        if z_sig - z <= ficha.espesor_repisa:
            avisos.append(f"{gid}: las repisas a {z} m y {z_sig} m no dejan hueco libre")
    return avisos


def descomponer_gondola(
    gid: str,
    poligono: Polygon,
    ficha: Ficha,
    caminable: Polygon,
    cfg: Config,
    avisos: list[str],
) -> list[Cara]:
    rect = poligono.minimum_rotated_rectangle
    if rect.area <= 0:
        avisos.append(f"{gid}: polígono degenerado, se ignora")
        return []
    if poligono.area / rect.area < 0.9:
        avisos.append(f"{gid}: no es rectangular ({poligono.area / rect.area:.0%} de su rectángulo envolvente); se usa el rectángulo")

    pts = list(orient(rect, sign=1.0).exterior.coords)[:4]
    lados = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        largo = math.dist(a, b)
        n = ((b[1] - a[1]) / largo, -(b[0] - a[0]) / largo)
        franja = _franja(a, b, n, cfg.prof_frente)
        frac = franja.intersection(caminable).area / franja.area
        lados.append((a, b, largo, n, frac))

    largo_max = max(l[2] for l in lados)
    es_largo = [l[2] >= largo_max - 1e-6 for l in lados]
    validas = [
        (es_largo[i] or ficha.caras_cortas) and l[2] >= cfg.largo_min_cara and l[4] >= cfg.frac_frente_caminable
        for i, l in enumerate(lados)
    ]

    caras: list[Cara] = []
    for i, (a, b, largo, n, frac) in enumerate(lados):
        if not validas[i]:
            continue
        # Si la cara opuesta también exhibe, cada una usa la mitad del fondo.
        fondo = lados[(i + 1) % 4][2]
        prof = min(fondo / (2 if validas[(i + 2) % 4] else 1), ficha.prof_max_repisa)
        cid = f"{gid}:{_cardinal(n)}"
        if any(c.id == cid for c in caras):
            cid = f"{cid}{i}"
        cara = Cara(cid, _cardinal(n), "largo" if es_largo[i] else "corto", a, b, n, largo, prof, frac)
        cara.tramos = _tramos(cid, a, b, largo, n, prof, ficha)
        caras.append(cara)

    if not caras:
        avisos.append(f"{gid}: ninguna cara da a un pasillo")
    return caras


def _franja(a: Punto, b: Punto, n: Punto, d: float) -> Polygon:
    return Polygon([a, b, (b[0] + n[0] * d, b[1] + n[1] * d), (a[0] + n[0] * d, a[1] + n[1] * d)])


def _cardinal(n: Punto) -> str:
    return CARDINALES[round(math.atan2(n[1], n[0]) / (math.pi / 2)) % 4]


def _tramos(cid: str, a: Punto, b: Punto, largo: float, n: Punto, prof: float, ficha: Ficha) -> list[Tramo]:
    n_tramos = max(1, round(largo / ficha.ancho_tramo))
    ancho = largo / n_tramos
    u = ((b[0] - a[0]) / largo, (b[1] - a[1]) / largo)
    adentro = (-n[0] * prof, -n[1] * prof)

    tramos = []
    for k in range(n_tramos):
        x0 = k * ancho
        q0 = (a[0] + u[0] * x0, a[1] + u[1] * x0)
        q1 = (a[0] + u[0] * (x0 + ancho), a[1] + u[1] * (x0 + ancho))
        huella = Polygon([q0, q1, (q1[0] + adentro[0], q1[1] + adentro[1]), (q0[0] + adentro[0], q0[1] + adentro[1])])
        tramo = Tramo(f"{cid}:{k + 1:02d}", k + 1, x0, ancho, q0, q1)
        for nivel, z in enumerate(ficha.niveles, 1):
            techo = ficha.niveles[nivel] if nivel < len(ficha.niveles) else ficha.alto
            hueco = techo - z - ficha.espesor_repisa
            if hueco > 0:
                tramo.repisas.append(Repisa(f"{tramo.id}:{nivel}", nivel, z, hueco, x0, ancho, prof, huella))
        tramos.append(tramo)
    return tramos
