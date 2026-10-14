"""Arma el grafo de la tienda a partir del plano, las fichas, las zonas y las reglas de categoría.

Cada nodo tiene el atributo `nodo` con su clase (tienda, gondola, cara, tramo,
repisa, pasillo, poi, zona, tipo_zona, categoria) y cada arista el atributo
`rel` con la relación (ver docs/arquitectura_layout.drawio, página 2).
La geometría se guarda como listas de coordenadas para poder exportar a JSON.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import networkx as nx
import pandas as pd
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..config import Config
from .estructura import FICHA_POR_DEFECTO, Ficha, descomponer_gondola, validar_ficha
from .lector_dxf import Plano
from .navmesh import Navmesh, construir_navmesh
from .zonas import Zona, asignar_zonas


def area_caminable(plano: Plano) -> Polygon:
    obstaculos = [*plano.gondolas.values(), *plano.cajas.values(), *plano.muros]
    return plano.contorno.difference(unary_union(obstaculos))


def construir_grafo(
    tienda_id: str,
    plano: Plano,
    fichas: dict[str, Ficha],
    zonas: list[Zona],
    categorias: pd.DataFrame,
    reglas: pd.DataFrame,
    cfg: Config = Config(),
) -> tuple[nx.DiGraph, list[str]]:
    avisos = list(plano.avisos)
    G = nx.DiGraph(tienda=tienda_id, tam_celda=cfg.tam_celda)
    tienda = f"S:{tienda_id}"
    G.add_node(tienda, nodo="tienda", contorno=_coords(plano.contorno), muros=[_coords(m) for m in plano.muros])

    caminable = area_caminable(plano)
    nav = construir_navmesh(caminable, cfg.tam_celda, cfg.frac_celda_caminable)
    _agregar_navmesh(G, nav, cfg)

    repisas = []
    for gid, poly in plano.gondolas.items():
        ficha = fichas.get(gid)
        if ficha is None:
            avisos.append(f"{gid}: no tiene fila en gondolas.csv; se usa la ficha por defecto ({FICHA_POR_DEFECTO.tipo})")
            ficha = FICHA_POR_DEFECTO
        avisos += validar_ficha(gid, ficha)
        caras = descomponer_gondola(gid, poly, ficha, caminable, cfg, avisos)

        g = f"G:{gid}"
        G.add_node(g, nodo="gondola", etiqueta=gid, tipo_gondola=ficha.tipo, alto=ficha.alto, poligono=_coords(poly))
        G.add_edge(tienda, g, rel="CONTIENE")
        for cara in caras:
            c = f"C:{cara.id}"
            G.add_node(c, nodo="cara", orientacion=cara.orientacion, lado=cara.lado, p0=cara.p0, p1=cara.p1,
                       normal=cara.normal, longitud=cara.longitud, profundidad=cara.profundidad,
                       frac_frente_caminable=cara.frac_frente_caminable)
            G.add_edge(g, c, rel="TIENE_CARA")
            _conectar_cara(G, c, cara.p0, cara.p1, cara.normal, nav, cfg, avisos)
            for tramo in cara.tramos:
                t = f"T:{tramo.id}"
                G.add_node(t, nodo="tramo", orden=tramo.orden, x0=tramo.x0, ancho=tramo.ancho, p0=tramo.p0, p1=tramo.p1)
                G.add_edge(c, t, rel="TIENE_TRAMO")
                for r in tramo.repisas:
                    G.add_node(f"R:{r.id}", nodo="repisa", nivel=r.nivel, z=r.z, hueco=r.hueco, x0=r.x0,
                               ancho=r.ancho, profundidad=r.profundidad, huella=_coords(r.huella))
                    G.add_edge(t, f"R:{r.id}", rel="TIENE_REPISA")
                    repisas.append(r)

    for z in zonas:
        G.add_node(f"Z:{z.id}", nodo="zona", etiqueta=z.id, tipo_zona=z.tipo, poligono=_coords(z.poligono),
                   z_min=z.z_min, z_max=z.z_max, exclusiva=z.exclusiva)
        G.add_edge(tienda, f"Z:{z.id}", rel="CONTIENE")
        G.add_node(f"TZ:{z.tipo}", nodo="tipo_zona", nombre=z.tipo)
        G.add_edge(f"Z:{z.id}", f"TZ:{z.tipo}", rel="ES_DE_TIPO")
    for rid, (zid, frac) in asignar_zonas(repisas, zonas, cfg, avisos).items():
        G.add_edge(f"R:{rid}", f"Z:{zid}", rel="DENTRO_DE", fraccion=frac)

    for clase, polys in (("entrada", plano.entradas), ("caja", plano.cajas)):
        for pid, poly in polys.items():
            p = f"POI:{pid}"
            G.add_node(p, nodo="poi", clase=clase, etiqueta=pid, poligono=_coords(poly))
            _conectar_poi(G, p, poly, nav, cfg, avisos)
    if not plano.entradas:
        avisos.append("El plano no tiene entrada (capa ENTRADA)")
    if not plano.cajas:
        avisos.append("El plano no tiene cajas (capa CAJA)")

    _agregar_categorias(G, categorias, reglas, avisos)
    _revisar_conectividad(G, avisos)
    return G, avisos


def _agregar_navmesh(G: nx.DiGraph, nav: Navmesh, cfg: Config) -> None:
    for c in nav.celdas.values():
        G.add_node(c.id, nodo="pasillo", i=c.i, j=c.j, centro=c.centro, area=c.area, holgura=c.holgura,
                   caja=tuple(c.poligono.bounds))
    for a, b, d in nav.vecinos:
        G.add_edge(a, b, rel="CONECTA", distancia=d)
        G.add_edge(b, a, rel="CONECTA", distancia=d)


def _celdas_en(poly: Polygon, nav: Navmesh) -> list[str]:
    return [nav.ids[k] for k in shapely.contains(poly, nav.puntos).nonzero()[0]]


def _celda_mas_cercana(geom, nav: Navmesh, max_dist: float) -> str | None:
    dist = shapely.distance(geom, nav.puntos)
    k = int(dist.argmin())
    return nav.ids[k] if dist[k] <= max_dist else None


def _conectar_cara(G, c, p0, p1, n, nav: Navmesh, cfg: Config, avisos: list[str]) -> None:
    d = cfg.alcance_cara_pasillo
    franja = Polygon([p0, p1, (p1[0] + n[0] * d, p1[1] + n[1] * d), (p0[0] + n[0] * d, p0[1] + n[1] * d)])
    segmento = shapely.LineString([p0, p1])
    celdas = _celdas_en(franja, nav)
    if not celdas:
        cercana = _celda_mas_cercana(franja, nav, 2 * d)
        if cercana is None:
            avisos.append(f"{c}: no hay pasillo frente a la cara")
            return
        avisos.append(f"{c}: pasillo angosto, se conecta a la celda más cercana")
        celdas = [cercana]
    for p in celdas:
        G.add_edge(c, p, rel="DA_A", distancia=segmento.distance(shapely.Point(nav.celdas[p].centro)))


def _conectar_poi(G, p, poly: Polygon, nav: Navmesh, cfg: Config, avisos: list[str]) -> None:
    celdas = _celdas_en(poly.buffer(cfg.alcance_cara_pasillo), nav)
    if not celdas:
        cercana = _celda_mas_cercana(poly, nav, 3 * cfg.alcance_cara_pasillo)
        if cercana is None:
            avisos.append(f"{p}: no queda junto a ningún pasillo")
            return
        celdas = [cercana]
    for c in celdas:
        G.add_edge(p, c, rel="UBICADO_EN")


def _agregar_categorias(G, categorias: pd.DataFrame, reglas: pd.DataFrame, avisos: list[str]) -> None:
    for f in categorias.itertuples(index=False):
        G.add_node(f"CAT:{f.categoria_id}", nodo="categoria", nombre=f.nombre, rol=f.rol)
    tipos_en_plano = {d["nombre"] for _, d in nodos(G, "tipo_zona")}
    for f in reglas.itertuples(index=False):
        tz = f"TZ:{f.tipo_zona}"
        if tz not in G:
            G.add_node(tz, nodo="tipo_zona", nombre=f.tipo_zona)
        G.add_edge(f"CAT:{f.categoria_id}", tz, rel="PERMITIDA_EN", obligatoria=bool(f.obligatoria))
        if f.obligatoria and f.tipo_zona not in tipos_en_plano:
            avisos.append(f"Categoría {f.categoria_id} solo puede ir en zonas {f.tipo_zona}, pero esta tienda no tiene ninguna")


def _revisar_conectividad(G: nx.DiGraph, avisos: list[str]) -> None:
    pasillos = G.subgraph(n for n, _ in nodos(G, "pasillo"))
    componentes = list(nx.weakly_connected_components(pasillos))
    if len(componentes) > 1:
        tamanos = sorted((len(c) for c in componentes), reverse=True)
        avisos.append(f"El área caminable tiene {len(componentes)} partes desconectadas (celdas: {tamanos[:5]}...)")


def nodos(G: nx.DiGraph, clase: str) -> Iterator[tuple[str, dict]]:
    return ((n, d) for n, d in G.nodes(data=True) if d["nodo"] == clase)


def hijos(G: nx.DiGraph, n: str, rel: str) -> list[str]:
    return [m for m in G.successors(n) if G.edges[n, m]["rel"] == rel]


def _coords(geom) -> list:
    if geom.geom_type == "Polygon":
        return [list(p) for p in geom.exterior.coords]
    if geom.geom_type == "MultiPolygon":
        return [[list(p) for p in g.exterior.coords] for g in geom.geoms]
    return [list(p) for p in geom.coords]


def guardar_grafo(G: nx.DiGraph, ruta: Path) -> None:
    datos = nx.node_link_data(G, edges="aristas")
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def cargar_grafo(ruta: Path) -> nx.DiGraph:
    return nx.node_link_graph(json.loads(ruta.read_text(encoding="utf-8")), edges="aristas")
