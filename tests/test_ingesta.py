from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box

from planogram.config import Config
from planogram.ingesta.compatibilidad import repisas_compatibles, zona_de
from planogram.ingesta.datos import cargar_categorias, cargar_fichas, cargar_zonas
from planogram.ingesta.estructura import FICHA_POR_DEFECTO, descomponer_gondola
from planogram.ingesta.grafo import cargar_grafo, construir_grafo, guardar_grafo, hijos, nodos
from planogram.ingesta.lector_dxf import leer_dxf
from planogram.ingesta.zonas import Zona, fraccion_dentro
from planogram.sintetico.catalogo import generar_catalogo
from planogram.sintetico.plano import ParamsPlano, generar_plano


@pytest.fixture(scope="module")
def grafo(tmp_path_factory) -> nx.DiGraph:
    datos: Path = tmp_path_factory.mktemp("datos")
    generar_catalogo(datos, np.random.default_rng(0), semanas=2)
    dir_tienda = datos / "tiendas" / "T"
    generar_plano(dir_tienda, ParamsPlano(n_pasillos=3, largo_gondola=8.0))

    plano = leer_dxf(dir_tienda / "plano.dxf")
    avisos: list[str] = []
    zonas = cargar_zonas(dir_tienda / "zonas.csv", plano.zonas, avisos)
    categorias, reglas = cargar_categorias(datos)
    G, avisos_grafo = construir_grafo("T", plano, cargar_fichas(dir_tienda / "gondolas.csv"), zonas, categorias, reglas)
    assert avisos + avisos_grafo == []
    return G


def caras_de(G, gid: str) -> list[str]:
    return hijos(G, f"G:{gid}", "TIENE_CARA")


def repisas_de(G, cara: str) -> list[str]:
    return [r for t in hijos(G, cara, "TIENE_TRAMO") for r in hijos(G, t, "TIENE_REPISA")]


def tipo_zona(G, repisa: str) -> str | None:
    z = zona_de(G, repisa)
    return G.nodes[z]["tipo_zona"] if z else None


def test_dxf_en_mm_se_lee_en_metros(grafo):
    largo = grafo.nodes["C:G-01:E"]["longitud"]
    assert largo == pytest.approx(8.0)


def test_caras_segun_lo_que_tienen_enfrente(grafo):
    # Doble cara en el centro, una sola cara contra la pared o pegada a la góndola.
    assert sorted(caras_de(grafo, "G-01")) == ["C:G-01:E", "C:G-01:O"]
    assert caras_de(grafo, "P-O") == ["C:P-O:E"]
    assert caras_de(grafo, "CB-01S") == ["C:CB-01S:S"]
    assert caras_de(grafo, "MF-01") == ["C:MF-01:S"]
    # Las islas exhiben también por sus extremos.
    assert len(caras_de(grafo, "IS-01")) == 4


def test_profundidad_de_repisa(grafo):
    assert grafo.nodes["C:G-01:E"]["profundidad"] == pytest.approx(0.5)  # mitad del fondo
    assert grafo.nodes["C:P-O:E"]["profundidad"] == pytest.approx(0.6)  # fondo completo
    assert grafo.nodes["C:MF-01:S"]["profundidad"] == pytest.approx(0.7)  # tope de la ficha


def test_tramos_de_izquierda_a_derecha_mirando_la_cara(grafo):
    # La cara este se mira desde el este: la izquierda es el sur (y menor).
    t1, t2 = (grafo.nodes[f"T:G-01:E:{k:02d}"]["p0"] for k in (1, 2))
    assert t1[1] < t2[1]


def test_contencion_en_zonas(grafo):
    for cara in caras_de(grafo, "MF-01"):
        assert {tipo_zona(grafo, r) for r in repisas_de(grafo, cara)} == {"REFRIGERADO"}
    for cara in caras_de(grafo, "IS-01"):
        assert {tipo_zona(grafo, r) for r in repisas_de(grafo, cara)} == {"CONGELADO"}

    seguridad = [r for r in repisas_de(grafo, "C:G-02:E") if tipo_zona(grafo, r) == "SEGURIDAD"]
    assert sorted(seguridad) == ["R:G-02:E:05:4", "R:G-02:E:05:5", "R:G-02:E:06:4", "R:G-02:E:06:5"]
    # La zona cubre solo media góndola: la cara oeste queda libre.
    assert all(tipo_zona(grafo, r) is None for r in repisas_de(grafo, "C:G-02:O"))


def test_compatibilidad(grafo):
    compat = {c.removeprefix("CAT:"): set(rs) for c, rs in repisas_compatibles(grafo).items()}
    tipos = lambda cat: {tipo_zona(grafo, r) for r in compat[cat]}  # noqa: E731
    assert tipos("LAC") == {"REFRIGERADO"}
    assert tipos("HEL") == {"CONGELADO"}
    assert tipos("LIC") == {"SEGURIDAD"}
    assert tipos("BEB") == {"REFRIGERADO", None}  # permitida en frío, pero no obligatoria
    assert tipos("SNA") == {None}


def test_navmesh_conexo_y_caras_conectadas(grafo):
    pasillos = grafo.subgraph(n for n, _ in nodos(grafo, "pasillo"))
    assert nx.is_weakly_connected(pasillos)
    for c, _ in nodos(grafo, "cara"):
        assert hijos(grafo, c, "DA_A"), c
    entrada = "POI:E-01"
    caja = next(n for n, d in nodos(grafo, "poi") if d["clase"] == "caja")
    origen, destino = hijos(grafo, entrada, "UBICADO_EN")[0], hijos(grafo, caja, "UBICADO_EN")[0]
    assert nx.has_path(pasillos, origen, destino)


def test_guardar_y_cargar(grafo, tmp_path):
    ruta = tmp_path / "grafo.json"
    guardar_grafo(grafo, ruta)
    G2 = cargar_grafo(ruta)
    assert G2.number_of_nodes() == grafo.number_of_nodes()
    assert G2.number_of_edges() == grafo.number_of_edges()


def test_cara_bloqueada_no_se_crea():
    gondola = box(0, 0, 4, 1)
    caminable = box(-5, -5, 10, 10).difference(box(-1, 1, 5, 3))  # algo pegado por el norte
    avisos: list[str] = []
    caras = descomponer_gondola("X", gondola, FICHA_POR_DEFECTO, caminable.difference(gondola), Config(), avisos)
    assert [c.orientacion for c in caras] == ["S"]
    assert caras[0].profundidad == pytest.approx(FICHA_POR_DEFECTO.prof_max_repisa)


def test_fraccion_dentro_parcial():
    caras = descomponer_gondola("X", box(0, 0, 2, 1), FICHA_POR_DEFECTO, box(-5, -5, 10, 10).difference(box(0, 0, 2, 1)), Config(), [])
    repisa = next(c for c in caras if c.orientacion == "S").tramos[0].repisas[0]  # z = 0.15, hueco = 0.38
    zona = Zona("Z", "X", box(0, 0, 0.5, 0.5), 0.0, 0.34, True)
    assert fraccion_dentro(repisa, zona) == pytest.approx(0.5 * (0.34 - 0.15) / 0.38)
