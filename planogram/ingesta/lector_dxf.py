"""Lee un DXF y devuelve la geometría de la tienda en metros, agrupada por capa.

Cada polígono cerrado de las capas GONDOLA, ZONA, CAJA y ENTRADA se identifica
con el texto (TEXT o MTEXT) de la misma capa que cae dentro de él. Si no hay
texto se le asigna un id automático y se deja un aviso.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..config import Config

# Código $INSUNITS del DXF → metros por unidad.
METROS_POR_UNIDAD = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}


@dataclass
class Plano:
    contorno: Polygon
    muros: list[BaseGeometry]
    gondolas: dict[str, Polygon]
    zonas: dict[str, Polygon]
    entradas: dict[str, Polygon]
    cajas: dict[str, Polygon]
    avisos: list[str] = field(default_factory=list)


def leer_dxf(ruta: Path, cfg: Config = Config()) -> Plano:
    doc = ezdxf.readfile(ruta)
    msp = doc.modelspace()
    avisos: list[str] = []

    unidades = doc.header.get("$INSUNITS", 0)
    escala = METROS_POR_UNIDAD.get(unidades)
    if escala is None:
        escala = cfg.escala_por_defecto
        avisos.append(f"El DXF no declara unidades ($INSUNITS={unidades}); se asume escala {escala} m/unidad")

    capas = cfg.capas
    cerrados: dict[str, list[Polygon]] = {}
    abiertos: dict[str, list[LineString]] = {}
    textos: dict[str, list[tuple[Point, str]]] = {}

    for e in msp:
        capa = e.dxf.layer.upper()
        tipo = e.dxftype()
        if tipo in ("LWPOLYLINE", "POLYLINE"):
            pts, cerrado = _puntos_polilinea(e, escala)
            if cerrado and len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                cerrados.setdefault(capa, []).append(poly)
            elif len(pts) >= 2:
                abiertos.setdefault(capa, []).append(LineString(pts))
        elif tipo == "LINE":
            a, b = e.dxf.start, e.dxf.end
            abiertos.setdefault(capa, []).append(LineString([(a.x * escala, a.y * escala), (b.x * escala, b.y * escala)]))
        elif tipo in ("TEXT", "MTEXT"):
            if tipo == "TEXT":
                # p2 solo existe en textos ALIGNED/FIT; ahí se usa el punto medio de la línea base.
                _, p1, p2 = e.get_placement()
                p = p1.lerp(p2) if p2 is not None else p1
                contenido = e.dxf.text
            else:
                p = e.dxf.insert
                contenido = e.plain_text()
            textos.setdefault(capa, []).append((Point(p.x * escala, p.y * escala), contenido.strip()))

    def poligonos(capa: str) -> list[Polygon]:
        return cerrados.get(capa.upper(), [])

    def etiquetados(capa: str, prefijo: str) -> dict[str, Polygon]:
        return _etiquetar(poligonos(capa), textos.get(capa.upper(), []), prefijo, capa, avisos)

    gondolas = etiquetados(capas.gondola, "G")
    zonas = etiquetados(capas.zona, "Z")
    entradas = etiquetados(capas.entrada, "E")
    cajas = etiquetados(capas.caja, "CJ")

    muros: list[BaseGeometry] = [p for p in poligonos(capas.muro)]
    muros += [l.buffer(cfg.espesor_muro / 2, cap_style="square") for l in abiertos.get(capas.muro.upper(), [])]

    contornos = poligonos(capas.contorno)
    if contornos:
        contorno = max(contornos, key=lambda p: p.area)
        if len(contornos) > 1:
            avisos.append(f"Capa {capas.contorno}: {len(contornos)} polígonos; se usa el de mayor área")
    else:
        todo = unary_union([*gondolas.values(), *muros, *cajas.values(), *entradas.values()])
        contorno = box(*todo.bounds)
        avisos.append(f"No hay polígono en la capa {capas.contorno}; se usa el rectángulo envolvente del dibujo")

    if not gondolas:
        avisos.append(f"No se encontraron góndolas en la capa {capas.gondola}")

    return Plano(contorno, muros, gondolas, zonas, entradas, cajas, avisos)


def _puntos_polilinea(e, escala: float) -> tuple[list[tuple[float, float]], bool]:
    if e.dxftype() == "LWPOLYLINE":
        pts = [(x * escala, y * escala) for x, y in e.get_points("xy")]
        cerrado = e.closed
    else:
        pts = [(v.dxf.location.x * escala, v.dxf.location.y * escala) for v in e.vertices]
        cerrado = e.is_closed
    # Polilíneas "cerradas a mano": el último punto repite el primero.
    if len(pts) > 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
        cerrado = True
    return pts, cerrado


def _etiquetar(
    poligonos: list[Polygon],
    textos: list[tuple[Point, str]],
    prefijo: str,
    capa: str,
    avisos: list[str],
) -> dict[str, Polygon]:
    resultado: dict[str, Polygon] = {}
    for i, poly in enumerate(poligonos, 1):
        dentro = [t for p, t in textos if t and poly.contains(p)]
        if not dentro:
            ident = f"{prefijo}-AUTO-{i:02d}"
            c = poly.centroid
            avisos.append(f"Capa {capa}: polígono sin etiqueta en ({c.x:.1f}, {c.y:.1f}); se nombra {ident}")
        else:
            ident = dentro[0]
            if len(dentro) > 1:
                avisos.append(f"Capa {capa}: polígono con varias etiquetas {dentro}; se usa {ident}")
        if ident in resultado:
            nuevo = f"{ident}#{i}"
            avisos.append(f"Capa {capa}: id {ident} repetido; el segundo se renombra {nuevo}")
            ident = nuevo
        resultado[ident] = poly
    return resultado
