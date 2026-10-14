"""Genera un plano de tienda en DXF que imita uno exportado desde AutoCAD.

El dibujo va en milímetros y solo contiene geometría y etiquetas de texto, como
un plano real. Las alturas y niveles de cada góndola, y el tipo y rango de altura
de cada zona, van en CSV aparte (gondolas.csv y zonas.csv).

Distribución (vista en planta, +y hacia el fondo de la tienda):

    ┌────────────── muebles de frío (zona REFRIGERADO) ──────────────┐
    │            islas de congelados (zona CONGELADO)                 │
    │ P-O   CB  CB  CB  CB      ← cabeceras                      P-E  │
    │ pared  G   G   G   G      ← góndolas de doble cara         pared│
    │        CB  CB  CB  CB                                           │
    │                                 CJ  CJ  CJ  ← cajas             │
    └──[ENTRADA]──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment

MURO = 0.2
PROF_GONDOLA = 1.0
ANCHO_PASILLO = 2.4
PROF_CABECERA = 0.5
PROF_PARED = 0.6
PROF_FRIO = 0.9
LARGO_ISLA, PROF_ISLA = 2.4, 1.0

FICHAS = {
    # tipo: (alto, niveles, ancho_tramo, prof_max_repisa, espesor_repisa, caras_cortas)
    "lineal": (2.1, "0.15;0.55;0.95;1.35;1.75", 1.0, 0.6, 0.02, 0),
    "cabecera": (1.8, "0.15;0.6;1.05;1.5", 1.0, 0.5, 0.02, 0),
    "mueble_frio": (2.1, "0.2;0.6;1.0;1.4;1.8", 1.25, 0.7, 0.03, 0),
    "isla": (0.9, "0.1;0.5", 1.2, 0.6, 0.02, 1),
}

COLOR_CAPA = {"CONTORNO": 8, "MURO": 7, "GONDOLA": 5, "ZONA": 6, "ENTRADA": 3, "CAJA": 1}


@dataclass(frozen=True)
class ParamsPlano:
    n_pasillos: int = 6
    largo_gondola: float = 12.0
    # Góndola con zona SEGURIDAD en las dos repisas superiores de dos tramos de su cara este.
    gondola_seguridad: int = 2


Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1) en metros


def generar_plano(dir_tienda: Path, params: ParamsPlano) -> None:
    dir_tienda.mkdir(parents=True, exist_ok=True)
    n, largo = params.n_pasillos, params.largo_gondola
    paso = PROF_GONDOLA + ANCHO_PASILLO
    x_ini = 4.0
    ancho = x_ini + (n - 1) * paso + PROF_GONDOLA + 4.0
    y_ini, y_fin = 8.0, 8.0 + largo
    y_isla = y_fin + PROF_CABECERA + 2.1
    alto = y_isla + PROF_ISLA + 3.3 + PROF_FRIO + MURO

    gondolas: list[tuple[str, str, Rect]] = []
    for k in range(n):
        gx = x_ini + k * paso
        gondolas.append((f"G-{k + 1:02d}", "lineal", (gx, y_ini, gx + PROF_GONDOLA, y_fin)))
        gondolas.append((f"CB-{k + 1:02d}S", "cabecera", (gx, y_ini - PROF_CABECERA, gx + PROF_GONDOLA, y_ini)))
        gondolas.append((f"CB-{k + 1:02d}N", "cabecera", (gx, y_fin, gx + PROF_GONDOLA, y_fin + PROF_CABECERA)))

    y_pared = (7.0, y_isla - 0.6)
    gondolas.append(("P-O", "lineal", (MURO, y_pared[0], MURO + PROF_PARED, y_pared[1])))
    gondolas.append(("P-E", "lineal", (ancho - MURO - PROF_PARED, y_pared[0], ancho - MURO, y_pared[1])))

    x_frio0, x_frio1 = 1.5, ancho - 1.5
    n_frio = math.ceil((x_frio1 - x_frio0) / 3.75)
    l_frio = (x_frio1 - x_frio0) / n_frio
    y_frio = (alto - MURO - PROF_FRIO, alto - MURO)
    for k in range(n_frio):
        x0 = x_frio0 + k * l_frio
        gondolas.append((f"MF-{k + 1:02d}", "mueble_frio", (x0, y_frio[0], x0 + l_frio, y_frio[1])))

    n_isla = max(1, int((ancho - 8.0) // (LARGO_ISLA + 1.6)))
    x_isla0 = (ancho - (n_isla * LARGO_ISLA + (n_isla - 1) * 1.6)) / 2
    islas = []
    for k in range(n_isla):
        x0 = x_isla0 + k * (LARGO_ISLA + 1.6)
        islas.append((x0, y_isla, x0 + LARGO_ISLA, y_isla + PROF_ISLA))
        gondolas.append((f"IS-{k + 1:02d}", "isla", islas[-1]))

    gx_seg = x_ini + (params.gondola_seguridad - 1) * paso
    zonas = [
        ("Z-REF", "REFRIGERADO", (x_frio0 - 0.1, y_frio[0] - 0.1, x_frio1 + 0.1, y_frio[1]), 0.0, 2.3, 1),
        ("Z-CONG", "CONGELADO", (islas[0][0] - 0.05, y_isla - 0.05, islas[-1][2] + 0.05, y_isla + PROF_ISLA + 0.05), 0.0, 1.0, 1),
        # Media profundidad de la góndola (solo cara este), tramos 5 y 6, repisas superiores.
        ("Z-SEG", "SEGURIDAD", (gx_seg + PROF_GONDOLA / 2, y_ini + 4.0, gx_seg + PROF_GONDOLA, y_ini + 6.0), 1.34, 2.1, 1),
    ]

    cajas = []
    x = ancho * 0.45
    while x + 0.9 < ancho - 1.5:
        cajas.append((f"CJ-{len(cajas) + 1:02d}", (x, 1.8, x + 0.9, 4.0)))
        x += 2.6
    entrada = ("E-01", (1.0, MURO, 4.0, 1.2))

    _escribir_dxf(dir_tienda / "plano.dxf", ancho, alto, gondolas, zonas, cajas, entrada)
    _escribir_fichas(dir_tienda / "gondolas.csv", gondolas)
    _escribir_zonas(dir_tienda / "zonas.csv", zonas)


def _mm(p: tuple[float, float]) -> tuple[float, float]:
    return (round(p[0] * 1000, 1), round(p[1] * 1000, 1))


def _esquinas(r: Rect) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = r
    return [_mm((x0, y0)), _mm((x1, y0)), _mm((x1, y1)), _mm((x0, y1))]


def _escribir_dxf(ruta, ancho, alto, gondolas, zonas, cajas, entrada) -> None:
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4  # milímetros
    for nombre, color in COLOR_CAPA.items():
        doc.layers.add(nombre, color=color)
    msp = doc.modelspace()

    def rect(r: Rect, capa: str, etiqueta: str | None = None) -> None:
        msp.add_lwpolyline(_esquinas(r), close=True, dxfattribs={"layer": capa})
        if etiqueta:
            centro = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
            texto = msp.add_text(etiqueta, height=250, dxfattribs={"layer": capa})
            texto.set_placement(_mm(centro), align=TextEntityAlignment.MIDDLE_CENTER)

    rect((0, 0, ancho, alto), "CONTORNO")
    m = MURO / 2
    muros = [
        ((m, m), (entrada[1][0], m)),  # frente, a la izquierda de la entrada
        ((entrada[1][2], m), (ancho - m, m)),  # frente, a la derecha de la entrada
        ((ancho - m, m), (ancho - m, alto - m)),
        ((ancho - m, alto - m), (m, alto - m)),
        ((m, alto - m), (m, m)),
    ]
    for a, b in muros:
        msp.add_line(_mm(a), _mm(b), dxfattribs={"layer": "MURO"})

    for gid, _, r in gondolas:
        rect(r, "GONDOLA", gid)
    for zid, _, r, *_ in zonas:
        rect(r, "ZONA", zid)
    for cid, r in cajas:
        rect(r, "CAJA", cid)
    rect(entrada[1], "ENTRADA", entrada[0])
    doc.saveas(ruta)


def _escribir_fichas(ruta, gondolas) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gondola_id", "tipo", "alto", "niveles", "ancho_tramo", "prof_max_repisa", "espesor_repisa", "caras_cortas"])
        for gid, tipo, _ in gondolas:
            w.writerow([gid, tipo, *FICHAS[tipo]])


def _escribir_zonas(ruta, zonas) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zona_id", "tipo", "z_min", "z_max", "exclusiva"])
        for zid, tipo, _, z_min, z_max, exclusiva in zonas:
            w.writerow([zid, tipo, z_min, z_max, exclusiva])
