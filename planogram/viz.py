"""Vistas para revisar el grafo: planta de la tienda y elevación (vista frontal) de caras."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle  # noqa: E402

from .ingesta.compatibilidad import zona_de  # noqa: E402
from .ingesta.grafo import hijos, nodos  # noqa: E402

COLOR_ZONA = {"REFRIGERADO": "#3b82f6", "CONGELADO": "#7c3aed", "SEGURIDAD": "#ea580c"}
COLOR_OTRA_ZONA = "#0d9488"
COLOR_LIBRE = "#e5e7eb"


def color_zona(tipo: str | None) -> str:
    if tipo is None:
        return COLOR_LIBRE
    return COLOR_ZONA.get(tipo, COLOR_OTRA_ZONA)


def plot_planta(G, ruta: Path) -> None:
    tienda = next(d for _, d in nodos(G, "tienda"))
    xs = [p[0] for p in tienda["contorno"]]
    ys = [p[1] for p in tienda["contorno"]]
    ancho, alto = max(xs) - min(xs), max(ys) - min(ys)
    fig, ax = plt.subplots(figsize=(min(18, 0.5 * ancho + 2), min(14, 0.5 * alto + 1.5)))

    celdas = [d["caja"] for _, d in nodos(G, "pasillo")]
    ax.add_collection(PolyCollection(
        [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in celdas],
        facecolors="#f1f5f9", edgecolors="#cbd5e1", linewidths=0.2, zorder=0,
    ))
    ax.plot(xs, ys, color="black", lw=1.5, zorder=1)
    for m in tienda["muros"]:
        ax.add_patch(MplPolygon(_anillo(m), closed=True, fc="#475569", ec="none", zorder=1))

    for _, d in nodos(G, "zona"):
        c = color_zona(d["tipo_zona"])
        ax.add_patch(MplPolygon(_anillo(d["poligono"]), closed=True, fc=c, ec=c, alpha=0.18, lw=0, zorder=2))
        ax.add_patch(MplPolygon(_anillo(d["poligono"]), closed=True, fill=False, ec=c, ls="--", lw=1.2, zorder=4))

    for g, d in nodos(G, "gondola"):
        anillo = _anillo(d["poligono"])
        ax.add_patch(MplPolygon(anillo, closed=True, fc="#d4d4d8", ec="#3f3f46", lw=0.8, zorder=3))
        cx = sum(p[0] for p in anillo) / len(anillo)
        cy = sum(p[1] for p in anillo) / len(anillo)
        alto_g = max(p[1] for p in anillo) - min(p[1] for p in anillo)
        ancho_g = max(p[0] for p in anillo) - min(p[0] for p in anillo)
        ax.text(cx, cy, d["etiqueta"], ha="center", va="center", fontsize=6, zorder=6,
                rotation=90 if alto_g > ancho_g * 1.5 else 0)

    # Caras: línea sobre el borde, marcas entre tramos y una flecha hacia el pasillo.
    lineas, marcas = [], []
    for c, d in nodos(G, "cara"):
        (x0, y0), (x1, y1), (nx_, ny_) = d["p0"], d["p1"], d["normal"]
        lineas.append([(x0 + nx_ * 0.04, y0 + ny_ * 0.04), (x1 + nx_ * 0.04, y1 + ny_ * 0.04)])
        for t in hijos(G, c, "TIENE_TRAMO"):
            px, py = G.nodes[t]["p0"]
            marcas.append([(px, py), (px + nx_ * 0.15, py + ny_ * 0.15)])
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.annotate("", xy=(mx + nx_ * 0.45, my + ny_ * 0.45), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->", color="#15803d", lw=0.8), zorder=5)
    ax.add_collection(LineCollection(lineas, colors="#15803d", linewidths=2, zorder=5))
    ax.add_collection(LineCollection(marcas, colors="#15803d", linewidths=0.8, zorder=5))

    for _, d in nodos(G, "poi"):
        c = "#16a34a" if d["clase"] == "entrada" else "#dc2626"
        ax.add_patch(MplPolygon(_anillo(d["poligono"]), closed=True, fc=c, ec="black", lw=0.6, alpha=0.8, zorder=3))
        anillo = _anillo(d["poligono"])
        ax.text(sum(p[0] for p in anillo) / len(anillo), sum(p[1] for p in anillo) / len(anillo),
                d["etiqueta"], ha="center", va="center", fontsize=6, color="white", zorder=6)

    tipos = sorted({d["tipo_zona"] for _, d in nodos(G, "zona")})
    leyenda = [
        Patch(fc="#d4d4d8", ec="#3f3f46", label="Góndola"),
        Patch(fc="#15803d", label="Cara (con tramos) → pasillo"),
        Patch(fc="#f1f5f9", ec="#cbd5e1", label="Celda de pasillo (navmesh)"),
        Patch(fc="#16a34a", label="Entrada"),
        Patch(fc="#dc2626", label="Caja"),
        *[Patch(fc=color_zona(t), alpha=0.35, ec=color_zona(t), ls="--", label=f"Zona {t}") for t in tipos],
    ]
    ax.legend(handles=leyenda, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    n = {k: sum(1 for _ in nodos(G, k)) for k in ("gondola", "cara", "tramo", "repisa", "pasillo")}
    ax.set_title(f"Tienda {G.graph['tienda']}: {n['gondola']} góndolas, {n['cara']} caras, "
                 f"{n['tramo']} tramos, {n['repisa']} repisas, {n['pasillo']} celdas de pasillo", fontsize=10)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def plot_elevaciones(G, caras: list[str], ruta: Path) -> None:
    """Vista frontal de cada cara, como la vería un cliente parado en el pasillo."""
    fig, axes = plt.subplots(len(caras), 1, figsize=(10, 2.6 * len(caras)), squeeze=False)
    for ax, c in zip(axes[:, 0], caras):
        d = G.nodes[c]
        gondola = next(iter(G.predecessors(c)))
        alto = G.nodes[gondola]["alto"]
        tipos = set()
        for t in hijos(G, c, "TIENE_TRAMO"):
            dt = G.nodes[t]
            ax.add_patch(Rectangle((dt["x0"], 0), dt["ancho"], alto, fill=False, ec="#71717a", lw=0.8))
            for r in hijos(G, t, "TIENE_REPISA"):
                dr = G.nodes[r]
                zona = zona_de(G, r)
                tipo = G.nodes[zona]["tipo_zona"] if zona else None
                tipos.add(tipo)
                ax.add_patch(Rectangle((dr["x0"] + 0.02, dr["z"]), dr["ancho"] - 0.04, dr["hueco"],
                                       fc=color_zona(tipo), ec="none", alpha=0.6 if tipo else 1))
                ax.plot([dr["x0"], dr["x0"] + dr["ancho"]], [dr["z"], dr["z"]], color="#27272a", lw=2)
                ax.text(dr["x0"] + 0.05, dr["z"] + dr["hueco"] / 2, f"{dr['nivel']}", fontsize=6, va="center")
            ax.text(dt["x0"] + dt["ancho"] / 2, -0.12, f"tramo {dt['orden']}", ha="center", va="top", fontsize=7)
        ax.set_xlim(-0.1, d["longitud"] + 0.1)
        ax.set_ylim(-0.3, alto + 0.1)
        ax.set_aspect("equal")
        ax.set_ylabel("z (m)")
        zonas_txt = ", ".join(sorted(t for t in tipos if t)) or "sin zona"
        ax.set_title(f"{c.removeprefix('C:')} (cara {d['lado']}, mira al {d['orientacion']}), "
                     f"profundidad de repisa {d['profundidad']:.2f} m · {zonas_txt}", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _anillo(coords) -> list:
    # MultiPolygon guardado como lista de anillos: se dibuja el primero.
    return coords[0] if coords and isinstance(coords[0][0], list) else coords
