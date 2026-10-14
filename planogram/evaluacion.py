"""Evaluación y validación de un planograma, independientes de cómo se generó."""

from __future__ import annotations

from collections import Counter

import networkx as nx
import pandas as pd

from .afinidad import peso_vecindad
from .config import Modelo
from .demanda import Sku, multiplicador, utilidad, venta
from .ingesta.compatibilidad import compatible, reglas_categorias
from .ingesta.grafo import nodos


def a_tabla(G: nx.DiGraph, ubicaciones, skus: dict[str, Sku], modelo: Modelo,
            frac_impulso: dict[str, float]) -> pd.DataFrame:
    """Planograma como tabla, con venta y utilidad esperadas según el tráfico que tiene el grafo ahora."""
    filas = []
    for ub in ubicaciones:
        s, r = skus[ub.sku], G.nodes[ub.repisa]
        m = multiplicador(r, frac_impulso.get(s.categoria, 0.0), modelo)
        tramo = next(iter(G.predecessors(ub.repisa)))
        cara = next(iter(G.predecessors(tramo)))
        filas.append({
            "gondola": next(iter(G.predecessors(cara))).removeprefix("G:"),
            "cara": cara.removeprefix("C:"),
            "tramo": G.nodes[tramo]["orden"],
            "nivel": r["nivel"],
            "repisa": ub.repisa.removeprefix("R:"),
            "x0": round(ub.x0, 3),
            "sku": s.id,
            "marca": s.marca,
            "categoria": s.categoria,
            "frentes": ub.frentes,
            "ancho_total": round(s.ancho * ub.frentes, 3),
            "venta_semanal": round(venta(s, ub.frentes, m, r["profundidad"], modelo), 2),
            "utilidad_semanal": round(utilidad(s, ub.frentes, m, r["profundidad"], modelo), 2),
        })
    return pd.DataFrame(filas)


def kpis(G: nx.DiGraph, plan: pd.DataFrame, skus: dict[str, Sku]) -> dict:
    capacidad = sum(d["ancho"] for _, d in nodos(G, "repisa"))
    return {
        "utilidad_semanal": round(float(plan["utilidad_semanal"].sum()), 0),
        "venta_semanal_unidades": round(float(plan["venta_semanal"].sum()), 0),
        "skus_listados": int(plan["sku"].nunique()),
        "skus_catalogo": len([s for s in skus.values() if s.demanda > 0]),
        "ocupacion": round(float(plan["ancho_total"].sum()) / capacidad, 3),
        "frentes_promedio": round(float(plan["frentes"].mean()), 2),
    }


def validar(G: nx.DiGraph, plan: pd.DataFrame, skus: dict[str, Sku]) -> list[str]:
    """Revisa todas las restricciones físicas y de zona. Devuelve las violaciones encontradas."""
    errores = []
    reglas = reglas_categorias(G)
    for sku, n in Counter(plan["sku"]).items():
        if n > 1:
            errores.append(f"{sku}: aparece {n} veces")
    for repisa, grupo in plan.groupby("repisa"):
        r = G.nodes[f"R:{repisa}"]
        # Con medidas exactas: la columna ancho_total viene redondeada al mm.
        ocupado = sum(skus[f.sku].ancho * f.frentes for f in grupo.itertuples(index=False))
        if ocupado > r["ancho"] + 1e-6:
            errores.append(f"{repisa}: ocupa {ocupado:.4f} m de {r['ancho']:.4f} m")
        tramos = grupo.sort_values("x0")
        fin = tramos["x0"] + tramos["ancho_total"]
        if (tramos["x0"].to_numpy()[1:] < fin.to_numpy()[:-1] - 1e-6).any():
            errores.append(f"{repisa}: hay SKUs superpuestos")
        for f in grupo.itertuples(index=False):
            s = skus[f.sku]
            if s.alto > r["hueco"] + 1e-9:
                errores.append(f"{repisa}/{f.sku}: alto {s.alto} m > hueco {r['hueco']:.3f} m")
            if s.prof > r["profundidad"] + 1e-9:
                errores.append(f"{repisa}/{f.sku}: fondo {s.prof} m > repisa {r['profundidad']:.3f} m")
            if not compatible(G, reglas[f"CAT:{s.categoria}"], f"R:{repisa}"):
                errores.append(f"{repisa}/{f.sku}: categoría {s.categoria} no permitida en esta repisa")
    return errores


def evaluar_afinidad(unidades, asignacion: dict[str, str], vecinos, afinidad: dict[tuple[str, str], float],
                     nombres: dict[str, str]) -> tuple[dict, pd.DataFrame]:
    """Puntaje de afinidad de una asignación y la lista de vecindades entre categorías distintas."""
    filas = []
    for a, b, tipo in vecinos:
        c, d = asignacion.get(unidades[a].id), asignacion.get(unidades[b].id)
        if not c or not d or c == d:
            continue
        filas.append({
            "unidad_a": unidades[a].id, "unidad_b": unidades[b].id, "tipo": tipo,
            "categoria_a": c, "categoria_b": d,
            "par": " / ".join(sorted([nombres[c], nombres[d]])),
            "peso": peso_vecindad(afinidad, c, d, tipo),
        })
    tabla = pd.DataFrame(filas, columns=["unidad_a", "unidad_b", "tipo", "categoria_a", "categoria_b", "par", "peso"])
    resumen = {
        "puntaje": round(float(tabla["peso"].sum()), 1),
        "bloques": contar_bloques(unidades, asignacion),
        "vecindades_afines": int((tabla["peso"] > 0).sum()),
        "vecindades_en_conflicto": int((tabla["peso"] < 0).sum()),
        "vecindades_neutras": int((tabla["peso"] == 0).sum()),
    }
    return resumen, tabla


def tabla_pasillos(unidades, asignacion: dict[str, str], vecinos, afinidad, nombres: dict[str, str]) -> pd.DataFrame:
    """Qué ve el cliente en cada pasillo: las categorías de las dos caras enfrentadas.

    Cada cara se lee de izquierda a derecha mirándola de frente, así que las dos
    secuencias de un mismo pasillo van en sentidos opuestos.
    """
    pasillos: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for a, b, tipo in vecinos:
        if tipo == "frente":
            pasillos.setdefault((unidades[a].cara, unidades[b].cara), set()).add((a, b))

    def secuencia(ks: set[int]) -> str:
        orden = sorted(ks, key=lambda k: (unidades[k].orden, unidades[k].zona is not None))
        cats = []
        for k in orden:
            c = asignacion.get(unidades[k].id, "-")
            if not cats or cats[-1] != c:
                cats.append(c)
        return " → ".join(nombres.get(c, c) for c in cats)

    filas = []
    for (cara_a, cara_b), pares in sorted(pasillos.items()):
        pesos = [peso_vecindad(afinidad, asignacion.get(unidades[a].id, ""), asignacion.get(unidades[b].id, ""), "frente")
                 for a, b in pares]
        filas.append({
            "cara_a": cara_a.removeprefix("C:"),
            "categorias_a": secuencia({a for a, _ in pares}),
            "cara_b": cara_b.removeprefix("C:"),
            "categorias_b": secuencia({b for _, b in pares}),
            "afinidad": round(sum(pesos), 1),
            "conflictos": sum(p < 0 for p in pesos),
        })
    return pd.DataFrame(filas)


def contar_bloques(unidades, asignacion: dict[str, str]) -> int:
    """Cantidad de bloques: tramos consecutivos de una cara con la misma categoría cuentan como uno.

    Mide fragmentación: si una categoría aparece en muchos bloques chicos, el cliente la encuentra repartida.
    """
    por_cara: dict[str, list] = {}
    for u in unidades:
        if u.zona is None:
            por_cara.setdefault(u.cara, []).append(u)
    bloques = 0
    for us in por_cara.values():
        anterior = None
        for u in sorted(us, key=lambda u: u.orden):
            c = asignacion.get(u.id)
            if c and c != anterior:
                bloques += 1
            anterior = c
    return bloques
