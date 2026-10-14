"""Visualizador 3D de una tienda en Blender.

Se ejecuta con el Python de Blender (solo usa la biblioteca estándar y bpy):

    blender --python herramientas/blender_tienda.py -- salida/T01 --datos data/sintetico
    blender -b --python herramientas/blender_tienda.py -- salida/T01 --render salida/T01/tienda_3d.png
    blender -b --python herramientas/blender_tienda.py -- salida/T01 --render vista.png --camara G-02:E
    blender --python herramientas/blender_tienda.py -- salida/T01 --version referencia

O más simple: python -m planogram blender --tienda T01

Lee grafo.json y planograma.csv de la carpeta de salida, y categorias.csv / skus.csv de --datos.
La escena queda en colecciones que se pueden prender y apagar: Estructura, Productos (un
objeto por categoría), Zonas, Etiquetas y Tráfico (apagada al inicio).
Unidades: metros, con z hacia arriba.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import bpy

# Paleta tab20 de matplotlib: los colores coinciden con los mapas 2D.
TAB20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
         "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
         "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
COLOR_ZONA = {"REFRIGERADO": "#3b82f6", "CONGELADO": "#7c3aed", "SEGURIDAD": "#ea580c"}
COLORES = {
    "piso": "#f4f4f5", "muro": "#64748b", "base": "#3f3f46", "panel": "#a1a1aa", "repisa": "#d4d4d8",
    "caja": "#dc2626", "entrada": "#16a34a", "texto": "#111827",
}
ALTO_MURO = 2.4
ESPESOR_REPISA = 0.025
VARIANTES_SKU = 3  # tonos por categoría para distinguir SKUs vecinos


# ---------------------------------------------------------------- colores y materiales

def hex_a_lineal(h: str, factor: float = 1.0) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(lin(min(1.0, c * factor)) for c in (r, g, b))


_materiales: dict[str, bpy.types.Material] = {}


def material(nombre: str, color: tuple[float, float, float], alpha: float = 1.0) -> bpy.types.Material:
    if nombre in _materiales:
        return _materiales[nombre]
    m = bpy.data.materials.new(nombre)
    m.diffuse_color = (*color, alpha)  # color en vista sólida y en Workbench
    m.use_backface_culling = True
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        if hasattr(m, "surface_render_method"):  # Blender 4.2+
            m.surface_render_method = "BLENDED"
        else:
            m.blend_method = "BLEND"
    _materiales[nombre] = m
    return m


# ---------------------------------------------------------------- geometría

class Capa:
    """Acumula cajas y prismas y los vuelca en un solo objeto (mucho más rápido que un objeto por caja)."""

    def __init__(self, nombre: str, coleccion: bpy.types.Collection):
        self.nombre, self.coleccion = nombre, coleccion
        self.verts: list[tuple[float, float, float]] = []
        self.caras: list[tuple[int, ...]] = []
        self.mats: list[str] = []

    def prisma(self, poligono: list[tuple[float, float]], z0: float, z1: float, mat: str) -> None:
        n = len(poligono)
        if _area(poligono) < 0:
            poligono = poligono[::-1]
        b = len(self.verts)
        self.verts += [(x, y, z0) for x, y in poligono] + [(x, y, z1) for x, y in poligono]
        self.caras.append(tuple(range(b + n - 1, b - 1, -1)))  # abajo
        self.caras.append(tuple(range(b + n, b + 2 * n)))  # arriba
        for i in range(n):
            j = (i + 1) % n
            self.caras.append((b + i, b + j, b + n + j, b + n + i))
        self.mats += [mat] * (n + 2)

    def plano(self, poligono: list[tuple[float, float]], z: float, mat: str) -> None:
        if _area(poligono) < 0:
            poligono = poligono[::-1]
        b = len(self.verts)
        self.verts += [(x, y, z) for x, y in poligono]
        self.caras.append(tuple(range(b, b + len(poligono))))
        self.mats.append(mat)

    def caja(self, origen, u, v, ancho: float, fondo: float, z0: float, alto: float, mat: str) -> None:
        """Caja con base en `origen`, `ancho` a lo largo de u y `fondo` a lo largo de v (vectores 2D unitarios)."""
        o = origen
        pts = [o, (o[0] + u[0] * ancho, o[1] + u[1] * ancho),
               (o[0] + u[0] * ancho + v[0] * fondo, o[1] + u[1] * ancho + v[1] * fondo),
               (o[0] + v[0] * fondo, o[1] + v[1] * fondo)]
        self.prisma(pts, z0, z0 + alto, mat)

    def crear(self, oculto: bool = False) -> bpy.types.Object | None:
        if not self.caras:
            return None
        malla = bpy.data.meshes.new(self.nombre)
        malla.from_pydata(self.verts, [], self.caras)
        nombres = sorted(set(self.mats))
        for n in nombres:
            malla.materials.append(_materiales[n])
        indice = {n: i for i, n in enumerate(nombres)}
        malla.polygons.foreach_set("material_index", [indice[m] for m in self.mats])
        malla.update()
        obj = bpy.data.objects.new(self.nombre, malla)
        self.coleccion.objects.link(obj)
        obj.hide_viewport = obj.hide_render = oculto
        return obj


def _area(p) -> float:
    return sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p))) / 2


def _anillos(coords) -> list[list[tuple[float, float]]]:
    """Polígono guardado como anillo cerrado, o MultiPolygon como lista de anillos."""
    anillos = coords if coords and isinstance(coords[0][0], list) else [coords]
    return [[(p[0], p[1]) for p in a[:-1]] if a[0] == a[-1] else [(p[0], p[1]) for p in a] for a in anillos]


def coleccion(nombre: str, oculta: bool = False) -> bpy.types.Collection:
    c = bpy.data.collections.new(nombre)
    bpy.context.scene.collection.children.link(c)
    if oculta:
        bpy.context.view_layer.layer_collection.children[nombre].hide_viewport = True
        c.hide_render = True
    return c


def texto(cuerpo: str, pos, normal, tamano: float, col: bpy.types.Collection) -> None:
    curva = bpy.data.curves.new(f"txt_{cuerpo}", type="FONT")
    curva.body = cuerpo
    curva.size = tamano
    curva.align_x, curva.align_y = "CENTER", "BOTTOM"
    obj = bpy.data.objects.new(f"Etiqueta {cuerpo}", curva)
    obj.location = pos
    # El texto queda vertical, mirando hacia el pasillo (en la dirección de la normal de la cara).
    obj.rotation_euler = (math.pi / 2, 0, math.atan2(normal[0], -normal[1]))
    obj.data.materials.append(_materiales["texto"])
    col.objects.link(obj)


# ---------------------------------------------------------------- escena

ARCHIVO_PLAN = {"optimizado": "planograma.csv", "referencia": "planograma_base.csv"}


def construir(dir_salida: Path, dir_datos: Path, version: str = "optimizado") -> dict:
    grafo = json.loads((dir_salida / "grafo.json").read_text(encoding="utf-8"))
    nodos = {n["id"]: n for n in grafo["nodes"]}
    hijos: dict[str, list[str]] = {}
    for a in grafo["aristas"]:
        hijos.setdefault(a["source"], []).append(a["target"])

    def de_clase(clase: str):
        return [n for n in grafo["nodes"] if n["nodo"] == clase]

    with open(dir_datos / "categorias.csv", encoding="utf-8") as f:
        categorias = {r["categoria_id"]: r["nombre"] for r in csv.DictReader(f)}
    with open(dir_datos / "skus.csv", encoding="utf-8") as f:
        skus = {r["sku_id"]: r for r in csv.DictReader(f)}
    with open(dir_salida / ARCHIVO_PLAN[version], encoding="utf-8") as f:
        plan = list(csv.DictReader(f))

    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for nombre, h in COLORES.items():
        material(nombre, hex_a_lineal(h))
    for k, cat in enumerate(sorted(categorias)):
        for v in range(VARIANTES_SKU):
            material(f"{cat}_{v}", hex_a_lineal(TAB20[k % 20], 0.85 + 0.15 * v))
    for tipo, h in COLOR_ZONA.items():
        material(f"zona_{tipo}", hex_a_lineal(h), alpha=0.15)

    col_est, col_prod = coleccion("Estructura"), coleccion("Productos")
    col_zonas, col_etiq, col_traf = coleccion("Zonas"), coleccion("Etiquetas"), coleccion("Tráfico", oculta=True)

    # Piso, muros, cajas y entrada
    tienda = de_clase("tienda")[0]
    est = Capa("Estructura", col_est)
    for anillo in _anillos(tienda["contorno"]):
        est.plano(anillo, 0.0, "piso")
    for muro in tienda["muros"]:
        for anillo in _anillos(muro):
            est.prisma(anillo, 0.0, ALTO_MURO, "muro")
    for poi in de_clase("poi"):
        for anillo in _anillos(poi["poligono"]):
            if poi["clase"] == "caja":
                est.prisma(anillo, 0.0, 0.9, "caja")
            else:
                est.plano(anillo, 0.004, "entrada")

    # Góndolas: zócalo, panel trasero de cada cara y repisas
    for g in de_clase("gondola"):
        caras = [nodos[c] for c in hijos.get(g["id"], []) if nodos[c]["nodo"] == "cara"]
        repisas = [nodos[r] for c in caras for t in hijos.get(c["id"], []) for r in hijos.get(t, [])
                   if nodos[r]["nodo"] == "repisa"]
        z_base = min((r["z"] for r in repisas), default=0.1) - ESPESOR_REPISA
        for anillo in _anillos(g["poligono"]):
            if z_base > 0.01:
                est.prisma(anillo, 0.0, z_base, "base")
        for c in caras:
            (x0, y0), (x1, y1), (nx, ny), d = c["p0"], c["p1"], c["normal"], c["profundidad"]
            atras = [(x0 - nx * d, y0 - ny * d), (x1 - nx * d, y1 - ny * d),
                     (x1 - nx * (d + 0.02), y1 - ny * (d + 0.02)), (x0 - nx * (d + 0.02), y0 - ny * (d + 0.02))]
            est.prisma(atras, 0.0, g["alto"], "panel")
        for r in repisas:
            est.prisma(_anillos(r["huella"])[0], r["z"] - ESPESOR_REPISA, r["z"], "repisa")
    est.crear()

    # Productos: una caja por frente, un objeto por categoría
    capas = {cat: Capa(f"Productos {cat} {nombre}", col_prod) for cat, nombre in categorias.items()}
    variante: dict[str, int] = {}
    n_cajas = 0
    for fila in plan:
        s = skus[fila["sku"]]
        repisa = nodos[f"R:{fila['repisa']}"]
        tramo = nodos["T:" + fila["repisa"].rsplit(":", 1)[0]]
        cara = nodos["C:" + fila["repisa"].rsplit(":", 2)[0]]
        (px, py), (qx, qy), (nx, ny) = tramo["p0"], tramo["p1"], cara["normal"]
        largo = math.hypot(qx - px, qy - py)
        u, v = ((qx - px) / largo, (qy - py) / largo), (-nx, -ny)
        ancho, alto, prof = float(s["ancho"]), float(s["alto"]), float(s["prof"])
        fondo = min(repisa["profundidad"], max(1, int(repisa["profundidad"] / prof + 1e-9)) * prof) - 0.01
        cat = s["categoria_id"]
        mat = f"{cat}_{variante.setdefault(fila['sku'], len(variante) % VARIANTES_SKU)}"
        for k in range(int(fila["frentes"])):
            x = float(fila["x0"]) + k * ancho
            origen = (px + u[0] * x + v[0] * 0.01, py + u[1] * x + v[1] * 0.01)
            capas[cat].caja(origen, u, v, ancho * 0.96, fondo, repisa["z"], alto, mat)
            n_cajas += 1
    for capa in capas.values():
        capa.crear()

    # Zonas como volúmenes semitransparentes
    for z in de_clase("zona"):
        capa = Capa(f"Zona {z['etiqueta']} {z['tipo_zona']}", col_zonas)
        if f"zona_{z['tipo_zona']}" not in _materiales:
            material(f"zona_{z['tipo_zona']}", hex_a_lineal("#0d9488"), alpha=0.15)
        for anillo in _anillos(z["poligono"]):
            capa.prisma(anillo, z["z_min"], z["z_max"], f"zona_{z['tipo_zona']}")
        obj = capa.crear()
        obj.show_wire = True

    # Etiquetas: el nombre de la categoría sobre cada bloque de tramos consecutivos de una cara
    cat_tramo: dict[str, dict[str, float]] = {}
    for fila in plan:
        t = "T:" + fila["repisa"].rsplit(":", 1)[0]
        cat = skus[fila["sku"]]["categoria_id"]
        cat_tramo.setdefault(t, {})
        cat_tramo[t][cat] = cat_tramo[t].get(cat, 0.0) + float(fila["ancho_total"])
    for c in de_clase("cara"):
        alto = nodos["G:" + c["id"][2:].rsplit(":", 1)[0]]["alto"]
        tramos = sorted((nodos[t] for t in hijos.get(c["id"], []) if nodos[t]["nodo"] == "tramo"),
                        key=lambda t: t["orden"])
        bloque: list[dict] = []
        for t in tramos + [None]:
            cat = max(cat_tramo[t["id"]], key=cat_tramo[t["id"]].get) if t and t["id"] in cat_tramo else None
            previa = max(cat_tramo[bloque[0]["id"]], key=cat_tramo[bloque[0]["id"]].get) if bloque else None
            if bloque and (t is None or cat != previa):
                (ax, ay), (bx, by) = bloque[0]["p0"], bloque[-1]["p1"]
                n = c["normal"]
                tamano = min(0.22, 0.9 * math.hypot(bx - ax, by - ay) / max(len(categorias[previa]), 1))
                texto(categorias[previa], ((ax + bx) / 2 + n[0] * 0.02, (ay + by) / 2 + n[1] * 0.02, alto + 0.05),
                      n, tamano, col_etiq)
                bloque = []
            if t is not None and cat is not None:
                bloque.append(t)

    # Tráfico simulado sobre el piso (colección apagada al inicio)
    traf = Capa("Tráfico", col_traf)
    celdas = [(n["caja"], n.get("trafico", 0.0)) for n in de_clase("pasillo")]
    t_max = max((t for _, t in celdas), default=1.0) or 1.0
    escala = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]
    for k, h in enumerate(escala):
        material(f"trafico_{k}", hex_a_lineal(h))
    for (x0, y0, x1, y1), t in celdas:
        k = min(len(escala) - 1, int(t / t_max * len(escala)))
        traf.plano([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], 0.006, f"trafico_{k}")
    traf.crear()

    xs = [p[0] for p in tienda["contorno"]]
    ys = [p[1] for p in tienda["contorno"]]
    return {"bbox": (min(xs), min(ys), max(xs), max(ys)), "nodos": nodos, "productos": n_cajas}


def preparar_vista(info: dict, camara: str | None) -> None:
    escena = bpy.context.scene
    x0, y0, x1, y1 = info["bbox"]
    cx, cy, ancho, largo = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0

    sol = bpy.data.objects.new("Sol", bpy.data.lights.new("Sol", type="SUN"))
    sol.data.energy = 3.0
    sol.rotation_euler = (math.radians(40), 0, math.radians(30))
    escena.collection.objects.link(sol)

    cam = bpy.data.objects.new("Cámara", bpy.data.cameras.new("Cámara"))
    escena.collection.objects.link(cam)
    escena.camera = cam
    objetivo = bpy.data.objects.new("Objetivo", None)
    escena.collection.objects.link(objetivo)
    if camara:  # como la ve un cliente que entra al pasillo: desde el extremo izquierdo de la cara
        c = info["nodos"][f"C:{camara}"]
        (ax, ay), (bx, by), (nx, ny) = c["p0"], c["p1"], c["normal"]
        ux, uy = (bx - ax) / c["longitud"], (by - ay) / c["longitud"]
        # 1 m hacia el pasillo (cabe en uno de 2 m) y 1,5 m antes del inicio de la cara.
        cam.location = (ax - ux * 1.5 + nx * 1.0, ay - uy * 1.5 + ny * 1.0, 1.6)
        objetivo.location = (ax + ux * 0.45 * c["longitud"], ay + uy * 0.45 * c["longitud"], 1.0)
        cam.data.lens = 18
    else:  # vista aérea en diagonal desde la entrada
        cam.location = (cx - 0.45 * ancho, cy - 0.95 * largo, 0.85 * max(ancho, largo))
        objetivo.location = (cx, cy + 0.05 * largo, 0.0)
        cam.data.lens = 32
    cam.data.clip_end = 500
    restriccion = cam.constraints.new("TRACK_TO")
    restriccion.target = objetivo
    restriccion.track_axis, restriccion.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"

    escena.render.engine = "BLENDER_WORKBENCH"
    sombreado = escena.display.shading
    sombreado.light, sombreado.color_type = "STUDIO", "MATERIAL"
    sombreado.show_shadows = sombreado.show_cavity = True
    # Las etiquetas son planas: sin esto, desde el pasillo de atrás se leen espejadas.
    # Toda la geometría tiene las normales hacia afuera, así que no se pierde nada más.
    sombreado.show_backface_culling = True
    escena.display.shadow_focus = 0.8
    if escena.world is None:
        escena.world = bpy.data.worlds.new("Mundo")
    escena.world.color = (0.9, 0.92, 0.95)

    # Vista 3D interactiva: sólido con colores de material, encuadrada en la tienda.
    for ventana in bpy.context.window_manager.windows:
        for area in ventana.screen.areas:
            if area.type != "VIEW_3D":
                continue
            espacio = area.spaces.active
            espacio.shading.type, espacio.shading.color_type = "SOLID", "MATERIAL"
            espacio.shading.show_shadows = espacio.shading.show_cavity = True
            espacio.shading.show_backface_culling = True
            espacio.clip_end = 500
            region = next(r for r in area.regions if r.type == "WINDOW")
            with bpy.context.temp_override(window=ventana, area=area, region=region):
                bpy.ops.view3d.view_camera()


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender_tienda")
    p.add_argument("salida", type=Path, help="carpeta con grafo.json y planograma.csv (ej. salida/T01)")
    p.add_argument("--datos", type=Path, default=Path("data/sintetico"))
    p.add_argument("--camara", help="id de una cara (ej. G-02:E) para mirarla desde el pasillo")
    p.add_argument("--render", type=Path, help="guarda una imagen PNG de la vista de la cámara")
    p.add_argument("--guardar", type=Path, help="guarda la escena como .blend")
    p.add_argument("--resolucion", type=int, nargs=2, default=(1600, 1000))
    p.add_argument("--version", choices=list(ARCHIVO_PLAN), default="optimizado",
                   help="planograma optimizado o el de la regla de referencia")
    args = p.parse_args(argv)

    info = construir(args.salida, args.datos, args.version)
    preparar_vista(info, args.camara)
    # El nombre de la escena dice qué versión es (se ve arriba a la derecha en Blender).
    bpy.context.scene.name = f"{args.salida.name} {args.version}"
    print(f"Escena lista ({args.version}): {info['productos']} frentes de producto")
    if args.render:
        escena = bpy.context.scene
        escena.render.resolution_x, escena.render.resolution_y = args.resolucion
        escena.render.filepath = str(args.render.resolve())
        bpy.ops.render.render(write_still=True)
        print(f"Imagen en {args.render}")
    if args.guardar:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.guardar.resolve()))
        print(f"Escena en {args.guardar}")


if __name__ == "__main__":
    main()
