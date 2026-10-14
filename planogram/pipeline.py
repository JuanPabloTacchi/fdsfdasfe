"""Corre todas las etapas para una tienda y compara contra el layout de referencia."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .afinidad import cargar_afinidad
from .config import Config, Modelo
from .demanda import cabe, cargar_skus, params_categorias
from .evaluacion import a_tabla, evaluar_afinidad, kpis, tabla_pasillos, validar
from .ingesta.compatibilidad import resumen_compatibilidad
from .ingesta.datos import cargar_categorias, cargar_fichas, cargar_zonas
from .ingesta.grafo import construir_grafo, guardar_grafo
from .ingesta.lector_dxf import leer_dxf
from .optim.base import macro_base, micro_base
from .optim.ciclo import optimizar_categorias
from .optim.macro import CurvaCategoria
from .optim.micro import resolver_micro
from .optim.surtido import curva, envolvente_concava, pasos_marginales
from .optim.unidades import actualizar_exposicion, adyacencias, construir_unidades, vecindades
from .salida import exportar_dxf, pdf_planogramas, plot_categorias, plot_trafico, png_cara
from .scoring import Simulador, aplicar_trafico
from .viz import plot_planta


def correr_tienda(dir_datos: Path, tienda: str, dir_salida: Path, cfg: Config = Config(),
                  modelo: Modelo = Modelo(), graficos: bool = True, descomponer: bool = False) -> dict:
    tiempos: dict[str, float] = {}
    dir_salida.mkdir(parents=True, exist_ok=True)
    dir_tienda = dir_datos / "tiendas" / tienda

    # Etapa 1: plano → grafo
    t0 = time.perf_counter()
    plano = leer_dxf(dir_tienda / "plano.dxf", cfg)
    avisos: list[str] = []
    zonas = cargar_zonas(dir_tienda / "zonas.csv", plano.zonas, avisos)
    categorias, reglas = cargar_categorias(dir_datos)
    nombres = dict(zip(categorias["categoria_id"], categorias["nombre"]))
    afinidad = cargar_afinidad(dir_datos / "afinidad_categorias.csv", set(nombres), avisos)
    G, avisos_grafo = construir_grafo(tienda, plano, cargar_fichas(dir_tienda / "gondolas.csv"), zonas,
                                      categorias, reglas, cfg)
    avisos += avisos_grafo
    tiempos["1_grafo"] = time.perf_counter() - t0

    # Etapa 2: tráfico de referencia (clientes que visitan tramos al azar) y visibilidad
    t0 = time.perf_counter()
    sim = Simulador(G, modelo)
    trafico_aleatorio, st_aleatorio = sim.aleatorio()
    referencia = aplicar_trafico(G, trafico_aleatorio, modelo)
    params = params_categorias(categorias, avisos)
    frac = {c: p.frac_impulso for c, p in params.items()}
    tiempos["2_trafico"] = time.perf_counter() - t0

    # Etapa 3a: categorías ↔ tráfico (ciclo hasta que el layout se estabiliza)
    t0 = time.perf_counter()
    skus = cargar_skus(dir_datos, tienda, modelo)
    unidades = construir_unidades(G, modelo.gamma_posicion)
    curvas = {}
    for cat in nombres:
        reps = [G.nodes[r] for u in unidades if cat in u.categorias for r in u.repisas]
        lista = [s for s in skus.values() if s.categoria == cat and s.demanda > 0 and any(cabe(s, r) for r in reps)]
        if not lista:
            avisos.append(f"Categoría {cat}: ningún SKU cabe en el espacio compatible de esta tienda")
            continue
        prof = statistics.median(r["profundidad"] for r in reps)
        w, v = curva(pasos_marginales(lista, prof, max(r["ancho"] for r in reps), modelo))
        s_ref = min(sum(s.ancho * s.frentes_ref for s in lista), w[-1])
        curvas[cat] = CurvaCategoria(envolvente_concava(w, v), rho=float(np.interp(s_ref, w, v)) / s_ref)
    vecinos = vecindades(G, unidades, modelo.ancho_max_pasillo)
    asignacion, trafico_opt, historial, stats_macro = optimizar_categorias(
        G, sim, unidades, curvas, adyacencias(unidades), vecinos, afinidad, params,
        trafico_aleatorio, referencia, modelo)
    tiempos["3a_macro"] = time.perf_counter() - t0

    # Etapa 3b: SKU → repisa, por categoría, con el tráfico que genera el layout elegido
    t0 = time.perf_counter()
    stats_micro: dict[str, dict] = {}

    def micro_opt(cat, repisas, lista, m=modelo):
        res, stats_micro[cat] = resolver_micro(G, repisas, lista, m, frac.get(cat, 0.0))
        return res

    def micro_ref(cat, repisas, lista):
        return micro_base(G, repisas, lista, modelo, frac.get(cat, 0.0))

    rapido = replace(modelo, tiempo_micro=min(modelo.tiempo_micro, 3.0))

    def micro_rapido(cat, repisas, lista):
        return resolver_micro(G, repisas, lista, rapido, frac.get(cat, 0.0))[0]

    ubicaciones = _micro_por_categoria(unidades, asignacion, skus, micro_opt)
    plan = a_tabla(G, ubicaciones, skus, modelo, frac)
    tiempos["3b_micro"] = time.perf_counter() - t0
    descomposicion = {}
    if descomponer:
        descomposicion["macro_opt+micro_base"] = _utilidad(
            G, _micro_por_categoria(unidades, asignacion, skus, micro_ref), skus, modelo, frac)

    # Referencia: reglas simples, evaluada con el tráfico que genera su propio layout
    asignacion_base = macro_base(G, unidades, skus)
    trafico_base, st_base = sim.por_listas(unidades, asignacion_base, params)
    aplicar_trafico(G, trafico_base, modelo, referencia)
    actualizar_exposicion(G, unidades)
    ubic_base = _micro_por_categoria(unidades, asignacion_base, skus, micro_ref)
    plan_base = a_tabla(G, ubic_base, skus, modelo, frac)
    if descomponer:
        descomposicion["macro_base+micro_opt"] = _utilidad(
            G, _micro_por_categoria(unidades, asignacion_base, skus, micro_rapido), skus, modelo, frac)
    escala_trafico = max(max(trafico_opt.values()), max(trafico_base.values()))
    if graficos:
        plot_trafico(G, dir_salida / "trafico_base.png", "regla de referencia", escala_trafico)
    aplicar_trafico(G, trafico_opt, modelo, referencia)
    actualizar_exposicion(G, unidades)
    _, st_opt = sim.por_listas(unidades, asignacion, params)

    # Etapa 4: evaluación y salidas
    t0 = time.perf_counter()
    errores, errores_base = validar(G, plan, skus), validar(G, plan_base, skus)
    k_opt, k_base = kpis(G, plan, skus), kpis(G, plan_base, skus)
    k_opt["recorrido_medio_m"], k_base["recorrido_medio_m"] = st_opt["recorrido_medio_m"], st_base["recorrido_medio_m"]
    por_cat = _comparar_categorias(plan, plan_base, unidades, asignacion, asignacion_base, nombres)
    af_opt, vec_opt = evaluar_afinidad(unidades, asignacion, vecinos, afinidad, nombres)
    af_base, vec_base = evaluar_afinidad(unidades, asignacion_base, vecinos, afinidad, nombres)

    plan.to_csv(dir_salida / "planograma.csv", index=False)
    plan_base.to_csv(dir_salida / "planograma_base.csv", index=False)
    por_cat.to_csv(dir_salida / "categorias.csv", index=False)
    pd.DataFrame(historial).to_csv(dir_salida / "iteraciones_trafico.csv", index=False)
    pd.DataFrame([{"unidad": u.id, "categoria": asignacion.get(u.id, ""), "metros": u.capacidad,
                   "exposicion": round(u.exposicion, 3), "visibilidad": round(u.visibilidad, 3)} for u in unidades]
                 ).to_csv(dir_salida / "asignacion_categorias.csv", index=False)
    resumen_compatibilidad(G).to_csv(dir_salida / "compatibilidad.csv", index=False)
    tabla_pasillos(unidades, asignacion, vecinos, afinidad, nombres).to_csv(dir_salida / "pasillos.csv", index=False)
    vec_opt[vec_opt["peso"] < 0].to_csv(dir_salida / "conflictos_afinidad.csv", index=False)
    (dir_salida / "avisos.txt").write_text("\n".join(avisos), encoding="utf-8")
    (dir_salida / "violaciones.txt").write_text("\n".join(errores) or "sin violaciones", encoding="utf-8")
    guardar_grafo(G, dir_salida / "grafo.json")
    exportar_dxf(dir_tienda / "plano.dxf", G, unidades, asignacion, dir_salida / "planograma.dxf")
    if graficos:
        plot_planta(G, dir_salida / "planta.png")
        plot_trafico(G, dir_salida / "trafico.png", "layout optimizado", escala_trafico)
        plot_categorias(G, unidades, asignacion, nombres, dir_salida / "categorias_optimizado.png",
                        f"Tienda {tienda}: categorías optimizadas", vec_opt)
        plot_categorias(G, unidades, asignacion_base, nombres, dir_salida / "categorias_base.png",
                        f"Tienda {tienda}: categorías con la regla de referencia", vec_base)
        pdf_planogramas(G, plan, skus, nombres, dir_salida / "planogramas.pdf")
        # Muestra: la cara con más utilidad, para revisar sin abrir el PDF.
        mejor = plan.groupby("cara")["utilidad_semanal"].sum().idxmax()
        png_cara(G, f"C:{mejor}", plan, skus, nombres, dir_salida / "planograma_muestra.png")
    tiempos["4_salida"] = time.perf_counter() - t0

    resultado = {
        "tienda": tienda,
        "optimizado": k_opt,
        "base": k_base,
        "mejora_utilidad": round(k_opt["utilidad_semanal"] / k_base["utilidad_semanal"] - 1, 4),
        "descomposicion": descomposicion,
        "afinidad": af_opt,
        "afinidad_base": af_base,
        "iteraciones_trafico": historial,
        "trafico_referencia": {"pasadas_por_tramo": round(referencia, 3), **st_aleatorio},
        "violaciones": len(errores),
        "violaciones_base": len(errores_base),
        "avisos": len(avisos),
        "macro": stats_macro,
        "micro": stats_micro,
        "tiempos_s": {k: round(v, 1) for k, v in tiempos.items()},
    }
    (dir_salida / "resumen.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")
    return resultado


def _utilidad(G, ubicaciones, skus, modelo, frac) -> float:
    return kpis(G, a_tabla(G, ubicaciones, skus, modelo, frac), skus)["utilidad_semanal"]


def _micro_por_categoria(unidades, asignacion: dict[str, str], skus, resolver) -> list:
    ubicaciones = []
    for cat in sorted(set(asignacion.values())):
        repisas = [r for u in unidades if asignacion.get(u.id) == cat for r in u.repisas]
        ubicaciones += resolver(cat, repisas, [s for s in skus.values() if s.categoria == cat and s.demanda > 0])
    return ubicaciones


def _comparar_categorias(plan, plan_base, unidades, asignacion, asignacion_base, nombres) -> pd.DataFrame:
    filas = []
    for cat, nombre in nombres.items():
        p, b = plan[plan["categoria"] == cat], plan_base[plan_base["categoria"] == cat]
        filas.append({
            "categoria": cat,
            "nombre": nombre,
            "metros_opt": round(sum(u.capacidad for u in unidades if asignacion.get(u.id) == cat), 1),
            "metros_base": round(sum(u.capacidad for u in unidades if asignacion_base.get(u.id) == cat), 1),
            "skus_opt": int(p["sku"].nunique()),
            "skus_base": int(b["sku"].nunique()),
            "utilidad_opt": round(float(p["utilidad_semanal"].sum()), 0),
            "utilidad_base": round(float(b["utilidad_semanal"].sum()), 0),
        })
    return pd.DataFrame(filas)
