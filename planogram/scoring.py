"""Etapa 2: tráfico de clientes y valor de cada repisa.

Dos formas de simular el tráfico (pasadas por celda del navmesh, por cliente):

- `Simulador.aleatorio`: cada cliente visita algunos tramos al azar. No depende de
  dónde quedan las categorías. Se usa como tráfico de **referencia**: la demanda
  histórica se observó con un layout "promedio".
- `Simulador.por_listas`: cada cliente arma una lista de compras (cada categoría
  con su `prob_lista`), camina a la ubicación más cercana de cada categoría de su
  lista en el orden más corto, hace alguna parada al azar y sale por la caja más
  cercana. Este tráfico **depende del layout**: si las categorías destino quedan
  lejos, la gente camina más y pasa frente a más productos.

`aplicar_trafico` escribe en cada repisa:

- `exposicion` = pasadas frente a su tramo / pasadas promedio del tráfico de referencia
  (con piso `piso_trafico`). La referencia queda fija: así un layout que hace caminar
  más genera más exposición en total, no solo la redistribuye.
- `visibilidad_rel` = visibilidad de la cara (cabecera, isla, cercanía a la entrada)
  × campana de altura, relativa al promedio de la tienda.

El modelo de demanda combina ambas según la fracción de impulso de cada categoría
(ver `demanda.multiplicador`).
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from .config import Modelo
from .demanda import ParamsCategoria
from .ingesta.grafo import hijos, nodos

Trafico = dict[str, float]  # celda de pasillo → pasadas por cliente


class Simulador:
    """Precalcula rutas entre los puntos que un cliente puede visitar: un ancla por tramo, entradas y cajas."""

    def __init__(self, G: nx.DiGraph, modelo: Modelo):
        self.G, self.modelo = G, modelo
        P = nx.Graph()
        for a, b, d in G.edges(data=True):
            if d["rel"] == "CONECTA":
                P.add_edge(a, b, distancia=d["distancia"])

        # Ancla de cada tramo: la celda de pasillo frente a su cara más cercana al centro del tramo.
        self.ancla_tramo: dict[str, str] = {}
        for c, _ in nodos(G, "cara"):
            celdas = hijos(G, c, "DA_A")
            for t in hijos(G, c, "TIENE_TRAMO"):
                if celdas:
                    medio = _punto_medio(G.nodes[t])
                    self.ancla_tramo[t] = min(celdas, key=lambda p: math.dist(G.nodes[p]["centro"], medio))
        self.entradas = [self._ancla_poi(p) for p, d in nodos(G, "poi") if d["clase"] == "entrada"]
        self.cajas = [self._ancla_poi(p) for p, d in nodos(G, "poi") if d["clase"] == "caja"]
        self.entradas = [a for a in self.entradas if a]
        self.cajas = [a for a in self.cajas if a]
        if not self.entradas or not self.cajas:
            raise ValueError("Para simular tráfico el plano necesita al menos una entrada y una caja")

        # Árbol de caminos mínimos desde cada ancla (predecesores; los caminos se arman al recorrer).
        anclas = set(self.ancla_tramo.values()) | set(self.entradas) | set(self.cajas)
        self.pred: dict[str, dict[str, list[str]]] = {}
        self.dist: dict[str, dict[str, float]] = {}
        for a in anclas:
            pred, dist = nx.dijkstra_predecessor_and_distance(P, a, weight="distancia")
            self.pred[a], self.dist[a] = pred, dist

        self.tramos = list(self.ancla_tramo)
        anchos = np.array([G.nodes[t]["ancho"] for t in self.tramos])
        self.prob_tramo = anchos / anchos.sum()
        self._visibilidad()

    def _ancla_poi(self, p: str) -> str | None:
        celdas = hijos(self.G, p, "UBICADO_EN")
        if not celdas:
            return None
        centro = _centro_poligono(self.G.nodes[p]["poligono"])
        return min(celdas, key=lambda c: math.dist(self.G.nodes[c]["centro"], centro))

    def _visibilidad(self) -> None:
        """Visibilidad de cada repisa: no depende del layout, se calcula una vez."""
        G, m = self.G, self.modelo
        d_entrada = self.dist[self.entradas[0]]
        dists = [d_entrada.get(a, math.inf) for a in self.ancla_tramo.values()]
        d_max = max((d for d in dists if math.isfinite(d)), default=1.0) or 1.0
        valores = {}
        for c, dc in nodos(G, "cara"):
            gondola = next(iter(G.predecessors(c)))
            base = {"cabecera": m.peso_cabecera, "isla": m.peso_isla}.get(G.nodes[gondola]["tipo_gondola"], 1.0)
            for t in hijos(G, c, "TIENE_TRAMO"):
                d = d_entrada.get(self.ancla_tramo.get(t, ""), math.inf)
                vis = base * (1 + m.bonus_cercania_entrada * (1 - d / d_max)) if math.isfinite(d) else base
                G.nodes[t]["visibilidad"] = vis
                for r in hijos(G, t, "TIENE_REPISA"):
                    dr = G.nodes[r]
                    dr["f_vertical"] = f_vertical(dr["z"] + dr["hueco"] / 2, m)
                    valores[r] = vis * dr["f_vertical"]
        media = float(np.mean(list(valores.values())))
        for r, v in valores.items():
            G.nodes[r]["visibilidad_rel"] = v / media

    # ------------------------------------------------------------ simulaciones

    def aleatorio(self) -> tuple[Trafico, dict]:
        """Cada cliente visita `paradas_por_compra` tramos al azar (proporcional al ancho)."""
        rng = np.random.default_rng(self.modelo.semilla)
        k = min(self.modelo.paradas_por_compra, len(self.tramos))
        destinos = []
        for _ in range(self.modelo.n_compradores):
            elegidos = rng.choice(len(self.tramos), size=k, replace=False, p=self.prob_tramo)
            destinos.append([[self.ancla_tramo[self.tramos[i]]] for i in elegidos])
        return self._simular(destinos, rng)

    def por_listas(self, unidades, asignacion: dict[str, str],
                   params: dict[str, ParamsCategoria]) -> tuple[Trafico, dict]:
        """Cada cliente va a buscar las categorías de su lista donde quedaron según `asignacion`."""
        rng = np.random.default_rng(self.modelo.semilla)
        # El cliente busca un producto concreto, que puede estar en cualquier parte del espacio de la
        # categoría: el tramo se sortea con probabilidad proporcional a sus metros. Ir siempre al tramo
        # más cercano de la categoría premiaría partirla en pedazos cerca de la entrada.
        anclas_cat: dict[str, list[str]] = {}
        metros_cat: dict[str, list[float]] = {}
        for u in unidades:
            c = asignacion.get(u.id)
            if c and u.tramo in self.ancla_tramo:
                anclas_cat.setdefault(c, []).append(self.ancla_tramo[u.tramo])
                metros_cat.setdefault(c, []).append(u.capacidad)
        cats = [c for c in anclas_cat if c in params]
        prob = np.array([params[c].prob_lista for c in cats])
        peso_tramo = {c: np.array(metros_cat[c]) / sum(metros_cat[c]) for c in cats}
        destinos = []
        for _ in range(self.modelo.n_compradores):
            lista = [c for c, p in zip(cats, prob) if rng.random() < p]
            if not lista:  # nadie entra sin buscar nada
                lista = [cats[rng.choice(len(cats), p=prob / prob.sum())]]
            paradas = [[anclas_cat[c][rng.choice(len(anclas_cat[c]), p=peso_tramo[c])]] for c in lista]
            for i in rng.choice(len(self.tramos), size=self.modelo.paradas_aleatorias, p=self.prob_tramo):
                paradas.append([self.ancla_tramo[self.tramos[i]]])
            destinos.append(paradas)
        return self._simular(destinos, rng)

    def _simular(self, destinos: list[list[list[str]]], rng) -> tuple[Trafico, dict]:
        """Cada cliente tiene una lista de paradas; cada parada se cumple en cualquiera de sus anclas
        (en `por_listas` cada parada tiene una sola).

        Ruta: desde la entrada, siempre a la parada pendiente más cercana (por su ancla más
        cercana); al final, a la caja más cercana.
        """
        conteo: dict[str, int] = {}
        largos = []
        for paradas in destinos:
            actual = self.entradas[rng.integers(len(self.entradas))]
            pendientes = [p for p in paradas if p]
            largo = 0.0
            while pendientes:
                d = self.dist[actual]
                i, ancla = min(((i, a) for i, p in enumerate(pendientes) for a in p),
                               key=lambda t: d.get(t[1], math.inf))
                largo += self._caminar(actual, ancla, conteo)
                actual = ancla
                pendientes.pop(i)
            caja = min(self.cajas, key=lambda a: self.dist[actual].get(a, math.inf))
            largo += self._caminar(actual, caja, conteo)
            largos.append(largo)
        n = len(destinos)
        trafico = {p: conteo.get(p, 0) / n for p, _ in nodos(self.G, "pasillo")}
        return trafico, {"recorrido_medio_m": round(float(np.mean(largos)), 1),
                         "paradas_medias": round(float(np.mean([len(p) for p in destinos])), 2)}

    def _caminar(self, desde: str, hasta: str, conteo: dict[str, int]) -> float:
        """Suma una pasada a cada celda del camino mínimo (sin contar la de partida). Devuelve el largo."""
        d = self.dist[desde].get(hasta, math.inf)
        if not math.isfinite(d):
            return 0.0
        pred = self.pred[desde]
        celda = hasta
        while celda != desde:
            conteo[celda] = conteo.get(celda, 0) + 1
            celda = pred[celda][0]
        return d


def mezclar(a: Trafico, b: Trafico, peso_b: float) -> Trafico:
    return {p: (1 - peso_b) * a.get(p, 0.0) + peso_b * b.get(p, 0.0) for p in a.keys() | b.keys()}


def trafico_tramos(G: nx.DiGraph, trafico: Trafico) -> dict[str, float]:
    """Pasadas promedio frente a cada tramo: celdas frente a su cara que caen dentro de su largo."""
    res = {}
    for c, d in nodos(G, "cara"):
        p0, p1 = np.array(d["p0"]), np.array(d["p1"])
        u = (p1 - p0) / d["longitud"]
        celdas = [(float(np.dot(np.array(G.nodes[p]["centro"]) - p0, u)), trafico.get(p, 0.0))
                  for p in hijos(G, c, "DA_A")]
        media_cara = float(np.mean([t for _, t in celdas])) if celdas else 0.0
        for t in hijos(G, c, "TIENE_TRAMO"):
            x0, ancho = G.nodes[t]["x0"], G.nodes[t]["ancho"]
            frente = [tr for x, tr in celdas if x0 - 0.25 <= x <= x0 + ancho + 0.25]
            res[t] = float(np.mean(frente)) if frente else media_cara
    return res


def aplicar_trafico(G: nx.DiGraph, trafico: Trafico, modelo: Modelo, referencia: float | None = None) -> float:
    """Escribe el tráfico en el grafo y la exposición de cada repisa. Devuelve la referencia usada.

    Si `referencia` es None se toma el promedio de pasadas por tramo de este tráfico
    (así se calibra con el tráfico aleatorio).
    """
    for p, _ in nodos(G, "pasillo"):
        G.nodes[p]["trafico"] = trafico.get(p, 0.0)
    por_tramo = trafico_tramos(G, trafico)
    if referencia is None:
        referencia = float(np.mean(list(por_tramo.values()))) or 1.0
    for t, pasadas in por_tramo.items():
        G.nodes[t]["trafico"] = pasadas
        expo = max(modelo.piso_trafico, pasadas / referencia)
        for r in hijos(G, t, "TIENE_REPISA"):
            dr = G.nodes[r]
            dr["exposicion"] = expo
            dr["score"] = expo * dr.get("visibilidad_rel", 1.0)  # para los mapas
    G.graph["trafico_referencia"] = referencia
    return referencia


def f_vertical(altura: float, modelo: Modelo) -> float:
    return modelo.piso_vertical + (1 - modelo.piso_vertical) * math.exp(
        -(((altura - modelo.altura_ojos) / modelo.ancho_vertical) ** 2))


def _punto_medio(d: dict) -> tuple[float, float]:
    return ((d["p0"][0] + d["p1"][0]) / 2, (d["p0"][1] + d["p1"][1]) / 2)


def _centro_poligono(coords) -> tuple[float, float]:
    xs, ys = [p[0] for p in coords[:-1]], [p[1] for p in coords[:-1]]
    return (sum(xs) / len(xs), sum(ys) / len(ys))
