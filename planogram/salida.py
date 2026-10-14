"""Etapa 4: mapas, planogramas por cara (PDF) y exportación a DXF."""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from ezdxf.enums import TextEntityAlignment  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle  # noqa: E402

from .demanda import Sku  # noqa: E402
from .ingesta.grafo import hijos, nodos  # noqa: E402
from .ingesta.lector_dxf import METROS_POR_UNIDAD  # noqa: E402
from .optim.unidades import Unidad  # noqa: E402
from .viz import _anillo  # noqa: E402


def colores_categoria(categorias: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {c: cmap(k % 20) for k, c in enumerate(sorted(categorias))}


def _base_planta(G, figsize_max=(16, 13)):
    tienda = next(d for _, d in nodos(G, "tienda"))
    xs = [p[0] for p in tienda["contorno"]]
    ys = [p[1] for p in tienda["contorno"]]
    ancho, alto = max(xs) - min(xs), max(ys) - min(ys)
    fig, ax = plt.subplots(figsize=(min(figsize_max[0], 0.45 * ancho + 3), min(figsize_max[1], 0.45 * alto + 1.5)))
    ax.plot(xs, ys, color="black", lw=1.2)
    for m in tienda["muros"]:
        ax.add_patch(MplPolygon(_anillo(m), closed=True, fc="#475569", ec="none"))
    for _, d in nodos(G, "gondola"):
        ax.add_patch(MplPolygon(_anillo(d["poligono"]), closed=True, fc="#e4e4e7", ec="#71717a", lw=0.6, zorder=2))
    for _, d in nodos(G, "poi"):
        c = "#16a34a" if d["clase"] == "entrada" else "#dc2626"
        ax.add_patch(MplPolygon(_anillo(d["poligono"]), closed=True, fc=c, ec="black", lw=0.5, zorder=3))
    ax.set_aspect("equal")
    return fig, ax


def plot_trafico(G, ruta: Path, etiqueta: str = "", t_max: float | None = None) -> None:
    """Tráfico por celda y score por tramo. `t_max` fija la escala de color, para comparar dos layouts."""
    fig, ax = _base_planta(G)
    celdas = [(d["caja"], d.get("trafico", 0.0)) for _, d in nodos(G, "pasillo")]
    t_max = t_max or max(t for _, t in celdas) or 1.0
    cmap = plt.get_cmap("YlOrRd")
    ax.add_collection(PolyCollection(
        [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for (x0, y0, x1, y1), _ in celdas],
        facecolors=[cmap(t / t_max) for _, t in celdas], edgecolors="none", zorder=1,
    ))
    # Franja sobre cada tramo con su score promedio.
    scores = {t: [G.nodes[r]["score"] for r in hijos(G, t, "TIENE_REPISA")] for t, _ in nodos(G, "tramo")}
    s_max = max(max(v) for v in scores.values() if v)
    cmap_s = plt.get_cmap("viridis")
    for t, vals in scores.items():
        if not vals:
            continue
        r0 = hijos(G, t, "TIENE_REPISA")[0]
        ax.add_patch(MplPolygon(G.nodes[r0]["huella"], closed=True, fc=cmap_s(sum(vals) / len(vals) / s_max),
                                ec="none", zorder=3))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t_max))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, label="pasadas por comprador")
    sm2 = plt.cm.ScalarMappable(cmap=cmap_s, norm=plt.Normalize(0, s_max))
    fig.colorbar(sm2, ax=ax, fraction=0.03, pad=0.04, label="score promedio del tramo")
    ax.set_title(f"Tienda {G.graph['tienda']}{' · ' + etiqueta if etiqueta else ''}: tráfico simulado "
                 f"(pasillos) y score de cada tramo (góndolas)", fontsize=10)
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def plot_categorias(G, unidades: list[Unidad], asignacion: dict[str, str], nombres: dict[str, str],
                    ruta: Path, titulo: str, conflictos: pd.DataFrame | None = None) -> None:
    fig, ax = _base_planta(G)
    colores = colores_categoria(list(nombres))
    for u in sorted(unidades, key=lambda u: u.zona is not None):
        cat = asignacion.get(u.id)
        huella = G.nodes[u.repisas[0]]["huella"]
        if u.zona:  # unidad dentro de una zona que parte el tramo: se dibuja con trama encima
            ax.add_patch(MplPolygon(huella, closed=True, fc=colores[cat] if cat else "white", ec="black",
                                    hatch="///", lw=0.4, zorder=4))
        else:
            ax.add_patch(MplPolygon(huella, closed=True, fc=colores[cat] if cat else "white", ec="white",
                                    lw=0.3, zorder=3))
    for _, d in nodos(G, "zona"):
        ax.add_patch(MplPolygon(_anillo(d["poligono"]), closed=True, fill=False, ec="black", ls="--", lw=0.8, zorder=5))

    # Etiqueta por bloque: tramos consecutivos de una cara con la misma categoría.
    por_cara: dict[str, list[Unidad]] = {}
    for u in unidades:
        if u.zona is None:
            por_cara.setdefault(u.cara, []).append(u)
    for cara, us in por_cara.items():
        us.sort(key=lambda u: u.orden)
        bloque: list[Unidad] = []
        for u in us + [None]:
            if bloque and (u is None or asignacion.get(u.id) != asignacion.get(bloque[0].id)):
                _etiquetar_bloque(ax, G, bloque, asignacion.get(bloque[0].id))
                bloque = []
            if u is not None:
                bloque.append(u)

    centro = {u.id: _centro_unidad(G, u) for u in unidades}
    if conflictos is not None:
        for f in conflictos[conflictos["peso"] < 0].itertuples(index=False):
            (x0, y0), (x1, y1) = centro[f.unidad_a], centro[f.unidad_b]
            ax.plot([x0, x1], [y0, y1], color="#dc2626", lw=1.2, zorder=6)

    usadas = sorted(set(asignacion.values()))
    leyenda = [Patch(fc=colores[c], label=f"{c} {nombres[c]}") for c in usadas]
    leyenda.append(Patch(fc="white", ec="black", label="sin asignar"))
    if conflictos is not None:
        leyenda.append(Line2D([], [], color="#dc2626", lw=1.2, label="vecinas que deberían separarse"))
    ax.legend(handles=leyenda, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_title(titulo, fontsize=10)
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _centro_unidad(G, u: Unidad) -> tuple[float, float]:
    huella = G.nodes[u.repisas[0]]["huella"][:-1]
    return (sum(p[0] for p in huella) / len(huella), sum(p[1] for p in huella) / len(huella))


def _etiquetar_bloque(ax, G, bloque: list[Unidad], cat: str | None) -> None:
    if not cat:
        return
    centros = [_centro_unidad(G, u) for u in bloque]
    x = sum(c[0] for c in centros) / len(centros)
    y = sum(c[1] for c in centros) / len(centros)
    normal = G.nodes[bloque[0].cara]["normal"]
    vertical = abs(normal[0]) > abs(normal[1])  # la cara mira al E u O: la góndola corre en y
    ax.text(x, y, cat, ha="center", va="center", fontsize=6, fontweight="bold", zorder=7,
            rotation=90 if vertical and len(bloque) > 1 else 0,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def figura_cara(G, c: str, plan: pd.DataFrame, skus: dict[str, Sku], nombres: dict[str, str]):
    """Vista frontal de una cara con cada frente de cada SKU, coloreado por categoría."""
    colores = colores_categoria(list(nombres))
    d = G.nodes[c]
    cid = c.removeprefix("C:")
    filas = plan[plan["cara"] == cid]
    alto = G.nodes[next(iter(G.predecessors(c)))]["alto"]
    fig, ax = plt.subplots(figsize=(max(6, min(22, 1.6 * d["longitud"] + 1)), 3.6))
    x_tramo, z_repisa = {}, {}
    for t in hijos(G, c, "TIENE_TRAMO"):
        dt = G.nodes[t]
        x_tramo[dt["orden"]] = dt["x0"]
        ax.add_patch(Rectangle((dt["x0"], 0), dt["ancho"], alto, fill=False, ec="#a1a1aa", lw=0.8))
        for r in hijos(G, t, "TIENE_REPISA"):
            dr = G.nodes[r]
            z_repisa[r.removeprefix("R:")] = dr["z"]
            ax.plot([dr["x0"], dr["x0"] + dr["ancho"]], [dr["z"], dr["z"]], color="#27272a", lw=1.5)
    for f in filas.itertuples(index=False):
        s = skus[f.sku]
        x, z = x_tramo[f.tramo] + f.x0, z_repisa[f.repisa]
        for k in range(f.frentes):
            ax.add_patch(Rectangle((x + k * s.ancho, z), s.ancho, s.alto, fc=colores[s.categoria], ec="white", lw=0.3))
        if f.ancho_total >= 0.12:
            ax.text(x + f.ancho_total / 2, z + s.alto / 2, f"{f.sku}\n×{f.frentes}", ha="center", va="center",
                    fontsize=3.5, rotation=90 if f.ancho_total < 0.25 else 0)
    cats = sorted(filas["categoria"].unique())
    ax.set_title(f"{cid} · {', '.join(nombres[k] for k in cats) or 'vacía'} · "
                 f"utilidad esperada ${filas['utilidad_semanal'].sum():,.0f}/semana", fontsize=8)
    ax.set_xlim(-0.05, d["longitud"] + 0.05)
    ax.set_ylim(-0.05, alto + 0.05)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def pdf_planogramas(G, plan: pd.DataFrame, skus: dict[str, Sku], nombres: dict[str, str], ruta: Path) -> None:
    with PdfPages(ruta) as pdf:
        for c, _ in sorted(nodos(G, "cara")):
            fig = figura_cara(G, c, plan, skus, nombres)
            pdf.savefig(fig)
            plt.close(fig)


def png_cara(G, c: str, plan: pd.DataFrame, skus: dict[str, Sku], nombres: dict[str, str], ruta: Path) -> None:
    fig = figura_cara(G, c, plan, skus, nombres)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def exportar_dxf(plano: Path, G, unidades: list[Unidad], asignacion: dict[str, str], ruta: Path) -> None:
    """Copia el plano y agrega la capa PLANOGRAMA con la categoría de cada tramo."""
    doc = ezdxf.readfile(plano)
    escala = METROS_POR_UNIDAD.get(doc.header.get("$INSUNITS", 0), 0.001)
    if "PLANOGRAMA" not in doc.layers:
        doc.layers.add("PLANOGRAMA", color=2)
    msp = doc.modelspace()
    for u in unidades:
        cat = asignacion.get(u.id)
        if not cat:
            continue
        cara, tramo = G.nodes[u.cara], G.nodes[u.tramo]
        (x0, y0), (x1, y1), (nx_, ny_) = tramo["p0"], tramo["p1"], cara["normal"]
        desplazamiento = 0.3 if u.zona is None else 0.6
        px = (x0 + x1) / 2 + nx_ * desplazamiento
        py = (y0 + y1) / 2 + ny_ * desplazamiento
        angulo = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if angulo > 90 or angulo <= -90:  # que el texto no quede de cabeza
            angulo -= 180 if angulo > 0 else -180
        texto = msp.add_text(cat if u.zona is None else f"{cat}*", height=0.15 / escala,
                             dxfattribs={"layer": "PLANOGRAMA", "rotation": angulo})
        texto.set_placement((px / escala, py / escala), align=TextEntityAlignment.MIDDLE_CENTER)
    doc.saveas(ruta)
