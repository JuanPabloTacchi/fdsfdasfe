from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from planogram.config import Modelo
from planogram.demanda import Sku, utilidad
from planogram.optim.surtido import curva, envolvente_concava, pasos_marginales, seleccionar
from planogram.pipeline import correr_tienda
from planogram.scoring import f_vertical
from planogram.sintetico.afinidad import generar_afinidad
from planogram.sintetico.catalogo import generar_catalogo
from planogram.sintetico.plano import ParamsPlano, generar_plano

RAPIDO = Modelo(n_compradores=100, tiempo_macro=5.0, tiempo_micro=1.0)


def sku(id="A", ancho=0.1, prof=0.1, demanda=30.0, margen=1.0, f_ref=2) -> Sku:
    return Sku(id, "X", "M", ancho, 0.2, prof, margen, demanda, f_ref)


def test_utilidad_crece_con_frentes_y_se_satura_por_stock():
    m = Modelo()
    s = sku(demanda=100.0)
    valores = [utilidad(s, f, 1.0, 0.5, m) for f in range(1, 9)]
    assert all(b >= a for a, b in zip(valores, valores[1:]))
    # Con 1 frente caben 5 unidades al fondo × 3 reposiciones = 15 por semana, menos que la demanda.
    assert valores[0] == pytest.approx(15.0)


def test_utilidad_cero_si_no_cabe_al_fondo():
    assert utilidad(sku(prof=0.6), 3, 1.0, 0.5, Modelo()) == 0.0


def test_mejor_ubicacion_vende_mas():
    m, s = Modelo(), sku(demanda=5.0)
    assert utilidad(s, 2, 1.2, 0.5, m) > utilidad(s, 2, 1.0, 0.5, m)


def test_f_vertical_maximo_a_la_altura_de_los_ojos():
    m = Modelo()
    assert f_vertical(m.altura_ojos, m) == pytest.approx(1.0)
    assert f_vertical(0.2, m) < f_vertical(0.9, m) < f_vertical(m.altura_ojos, m)


def test_analisis_marginal():
    m = Modelo()
    rng = np.random.default_rng(0)
    skus = [sku(f"S{k}", ancho=rng.uniform(0.05, 0.2), demanda=rng.uniform(1, 50)) for k in range(30)]
    pasos = pasos_marginales(skus, 0.5, 1.0, m)
    w, v = curva(pasos)
    assert (np.diff(w) > 0).all() and (np.diff(v) >= 0).all()
    hull = envolvente_concava(w, v)
    pendientes = np.diff([p[1] for p in hull]) / np.diff([p[0] for p in hull])
    assert (np.diff(pendientes) <= 1e-9).all()  # cóncava

    frentes = seleccionar(pasos, 2.0)
    assert sum(next(s for s in skus if s.id == k).ancho * f for k, f in frentes.items()) <= 2.0 + 1e-9
    assert all(1 <= f <= m.max_frentes for f in frentes.values())


@pytest.fixture(scope="module")
def resultado(tmp_path_factory) -> tuple[dict, Path]:
    datos: Path = tmp_path_factory.mktemp("datos")
    generar_catalogo(datos, np.random.default_rng(1), semanas=4)
    generar_afinidad(datos)
    # Catálogo reducido para que el test sea rápido.
    skus = pd.read_csv(datos / "skus.csv")
    skus.groupby("categoria_id").head(12).to_csv(datos / "skus.csv", index=False)
    pd.read_csv(datos / "tiendas.csv").assign(tienda_id="T").head(1).to_csv(datos / "tiendas.csv", index=False)
    ventas = pd.read_csv(datos / "ventas.csv")
    ventas[ventas["tienda_id"] == "T01"].assign(tienda_id="T").to_csv(datos / "ventas.csv", index=False)
    generar_plano(datos / "tiendas" / "T", ParamsPlano(n_pasillos=2, largo_gondola=6.0))
    salida = tmp_path_factory.mktemp("salida")
    return correr_tienda(datos, "T", salida, modelo=RAPIDO), salida


def test_pipeline_respeta_restricciones(resultado):
    r, _ = resultado
    assert r["violaciones"] == 0
    assert r["violaciones_base"] == 0


def test_pipeline_no_es_peor_que_la_referencia(resultado):
    r, _ = resultado
    assert r["optimizado"]["utilidad_semanal"] >= r["base"]["utilidad_semanal"]


def test_categorias_obligatorias_en_su_zona(resultado):
    _, salida = resultado
    plan = pd.read_csv(salida / "planograma.csv")
    assert set(plan[plan["categoria"] == "LAC"]["gondola"].str[:2]) <= {"MF"}
    assert set(plan[plan["categoria"].isin(["CON", "HEL"])]["gondola"].str[:2]) <= {"IS"}
    assert set(plan[plan["categoria"] == "LIC"]["repisa"]) <= {"G-02:E:05:4", "G-02:E:05:5", "G-02:E:06:4", "G-02:E:06:5"}


def test_salidas(resultado):
    _, salida = resultado
    for archivo in ["planograma.csv", "planograma_base.csv", "categorias.csv", "asignacion_categorias.csv",
                    "resumen.json", "planograma.dxf", "grafo.json", "planogramas.pdf", "planograma_muestra.png",
                    "trafico.png", "categorias_optimizado.png"]:
        assert (salida / archivo).exists(), archivo


def test_afinidad_mejora_sobre_la_referencia(resultado):
    r, _ = resultado
    assert r["afinidad"]["vecindades_en_conflicto"] <= r["afinidad_base"]["vecindades_en_conflicto"]
    assert r["afinidad"]["puntaje"] >= r["afinidad_base"]["puntaje"]


def test_tabla_de_pasillos(resultado):
    _, salida = resultado
    pasillos = pd.read_csv(salida / "pasillos.csv")
    # Las caras enfrentadas entre G-01 y G-02 forman un pasillo.
    assert ((pasillos["cara_a"] == "G-01:E") & (pasillos["cara_b"] == "G-02:O")).any()


def test_ciclo_elige_la_mejor_iteracion(resultado):
    r, _ = resultado
    historial = r["iteraciones_trafico"]
    assert historial and sum(h["elegida"] for h in historial) == 1
    elegida = next(h for h in historial if h["elegida"])
    assert elegida["valor_estimado"] == max(h["valor_estimado"] for h in historial)


def test_el_trafico_depende_de_donde_quedan_las_categorias(resultado):
    from planogram.demanda import ParamsCategoria
    from planogram.ingesta.grafo import cargar_grafo
    from planogram.optim.unidades import construir_unidades
    from planogram.scoring import Simulador

    _, salida = resultado
    G = cargar_grafo(salida / "grafo.json")
    unidades = construir_unidades(G, RAPIDO.gamma_posicion)
    sim = Simulador(G, RAPIDO)
    params = {"LAC": ParamsCategoria(prob_lista=1.0, frac_impulso=0.1)}
    # Todos buscan lácteos: si están junto a la entrada caminan menos que si están al fondo.
    cerca = {u.id: "LAC" for u in unidades if u.cara == "C:P-O:E" and u.orden <= 2}
    lejos = {u.id: "LAC" for u in unidades if u.cara.startswith("C:MF-")}
    _, st_cerca = sim.por_listas(unidades, cerca, params)
    _, st_lejos = sim.por_listas(unidades, lejos, params)
    assert st_lejos["recorrido_medio_m"] > st_cerca["recorrido_medio_m"]
