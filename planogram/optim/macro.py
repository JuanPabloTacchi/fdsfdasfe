"""Etapa 3a: qué categoría va en cada unidad de espacio (CP-SAT).

    max  Σ_c V_c(metros_c)                                  utilidad por espacio (curva cóncava)
       + Σ_u,c x[u,c] · ρ_c · capacidad_u · (mult_u,c − 1)  premio por ubicación (según tráfico e impulso de c)
       + Σ_(u,v) vecinas, c  y[u,v,c] · bono_c               bloques contiguos
       + Σ_(u,v) vecinas, c≠d  z[u,c,v,d] · afinidad(c,d) · k  afines enfrentadas, opuestas separadas
       − Σ_cara,c  presente[cara,c] · costo                  pocas categorías por cara
       − Σ_u  vacia[u] · costo_vacia                          no dejar repisas vacías

    s.a. cada unidad recibe a lo más una categoría, y solo una compatible.

- V_c se aproxima con la envolvente cóncava de la curva del análisis marginal,
  escrita como el mínimo de sus rectas (v_c ≤ a_k + b_k · metros_c).
- ρ_c es la utilidad por metro de la categoría: el término de ubicación es la
  expansión de primer orden de V_c · multiplicador alrededor de multiplicador = 1.
CP-SAT trabaja con enteros: metros en cm y utilidad en centavos.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ortools.sat.python import cp_model

from ..config import Modelo
from ..afinidad import peso_vecindad
from .unidades import Unidad

CM = 100
CENT = 100
ESCALA_PENDIENTE = 1000


@dataclass
class CurvaCategoria:
    puntos: list[tuple[float, float]]  # envolvente cóncava (metros, utilidad semanal)
    rho: float  # utilidad por metro en el espacio de referencia


def resolver_macro(
    unidades: list[Unidad],
    curvas: dict[str, CurvaCategoria],
    pares: list[tuple[int, int]],
    modelo: Modelo,
    vecinos: list[tuple[int, int, str]] = (),
    afinidad: dict[tuple[str, str], float] | None = None,
    frac_impulso: dict[str, float] | None = None,
    pista: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict]:
    """`pista`: asignación de partida (unidad → categoría), por ejemplo la de la iteración anterior."""
    frac_impulso = frac_impulso or {}
    m = cp_model.CpModel()
    x: dict[tuple[int, str], cp_model.IntVar] = {}
    for k, u in enumerate(unidades):
        for c in u.categorias:
            if c in curvas:
                x[k, c] = m.new_bool_var(f"x_{k}_{c}")

    objetivo = []
    vacias: list[cp_model.IntVar] = []
    for k, u in enumerate(unidades):
        opciones = [x[k, c] for c in u.categorias if (k, c) in x]
        if opciones:
            vacia = m.new_bool_var(f"vacia_{k}")
            m.add_exactly_one(opciones + [vacia])
            vacias.append(vacia)
    for c, curva in curvas.items():
        terminos = [(x[k, c], round(u.capacidad * CM)) for k, u in enumerate(unidades) if (k, c) in x]
        if not terminos:
            continue
        metros = m.new_int_var(0, sum(cap for _, cap in terminos), f"metros_{c}")
        m.add(metros == sum(var * cap for var, cap in terminos))
        v_max = round(curva.puntos[-1][1] * CENT)
        valor = m.new_int_var(0, v_max, f"valor_{c}")
        for (w0, v0), (w1, v1) in zip(curva.puntos, curva.puntos[1:]):
            if w1 <= w0:
                continue
            # valor ≤ v0 + pendiente · (metros − w0), escalado para usar coeficientes enteros
            pendiente = round((v1 - v0) * CENT / ((w1 - w0) * CM) * ESCALA_PENDIENTE)
            m.add(ESCALA_PENDIENTE * valor <= round(v0 * CENT * ESCALA_PENDIENTE) + pendiente * (metros - round(w0 * CM)))
        objetivo.append(valor)
        # No tiene sentido dar más metros de los que la categoría puede llenar.
        m.add(metros <= round(curva.puntos[-1][0] * CM))

        # Premio por ubicación, acotado: no puede superar (mult_max − 1) × lo que vende la categoría.
        # Sin esta cota, la categoría con más utilidad por metro acapara las mejores unidades aunque no las llene.
        # Solo se premia la parte sobre el promedio: si las unidades malas restaran, el modelo prefiere
        # dejarlas vacías, aunque en la práctica cualquier producto ahí vende algo.
        alfa = frac_impulso.get(c, 0.0)
        premios = [(x[k, c], round(curva.rho * u.capacidad * max(u.multiplicador(alfa) - 1, 0) * CENT))
                   for k, u in enumerate(unidades) if (k, c) in x]
        mult_max = max(u.multiplicador(alfa) for k, u in enumerate(unidades) if (k, c) in x)
        premio = m.new_int_var(0, sum(p for _, p in premios), f"premio_{c}")
        m.add(premio <= sum(var * p for var, p in premios))
        m.add(100 * premio <= round(100 * max(mult_max - 1, 0)) * valor)
        objetivo.append(premio)

    for a, b in pares:
        for c in set(unidades[a].categorias) & set(unidades[b].categorias):
            if (a, c) in x and (b, c) in x:
                bono = round(modelo.bonus_contiguidad * curvas[c].rho
                             * (unidades[a].capacidad + unidades[b].capacidad) / 2 * CENT)
                if bono <= 0:
                    continue
                y = m.new_bool_var(f"y_{a}_{b}_{c}")
                m.add_implication(y, x[a, c])
                m.add_implication(y, x[b, c])
                objetivo.append(bono * y)

    # Afinidad: z = x[u,c] ∧ x[v,d]. Con peso positivo basta z ≤ cada x (se maximiza);
    # con peso negativo hay que forzar z = 1 cuando ambas están puestas.
    n_afinidad = 0
    if afinidad and curvas:
        escala = modelo.peso_afinidad * statistics.mean(c.rho for c in curvas.values())
        for a, b, tipo in vecinos:
            cap = (unidades[a].capacidad + unidades[b].capacidad) / 2
            for c in unidades[a].categorias:
                for d in unidades[b].categorias:
                    coef = round(peso_vecindad(afinidad, c, d, tipo) * escala * cap * CENT)
                    if not coef or (a, c) not in x or (b, d) not in x:
                        continue
                    z = m.new_bool_var(f"z_{a}_{c}_{b}_{d}")
                    if coef > 0:
                        m.add_implication(z, x[a, c])
                        m.add_implication(z, x[b, d])
                    else:
                        m.add_bool_or([x[a, c].Not(), x[b, d].Not(), z])
                    objetivo.append(coef * z)
                    n_afinidad += 1

    # Costo por categoría presente en cada cara y por unidad vacía.
    if curvas:
        cap_media = statistics.mean(u.capacidad for u in unidades)
        valor_tramo = statistics.mean(c.rho for c in curvas.values()) * cap_media * CENT
        objetivo += [-round(modelo.costo_unidad_vacia * valor_tramo) * v for v in vacias]
        costo = round(modelo.costo_categoria_por_cara * statistics.mean(c.rho for c in curvas.values()) * cap_media * CENT)
        por_cara: dict[tuple[str, str], list] = {}
        for (k, c), var in x.items():
            por_cara.setdefault((unidades[k].cara, c), []).append(var)
        for (cara, c), vars_ in por_cara.items():
            presente = m.new_bool_var(f"p_{cara}_{c}")
            for var in vars_:
                m.add_implication(var, presente)
            objetivo.append(-costo * presente)

    if pista:
        for (k, c), var in x.items():
            m.add_hint(var, int(pista.get(unidades[k].id) == c))

    m.maximize(sum(objetivo))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = modelo.tiempo_macro
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = modelo.semilla
    estado = solver.solve(m)
    if estado not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"El nivel macro no encontró solución ({solver.status_name(estado)})")

    asignacion = {unidades[k].id: c for (k, c), var in x.items() if solver.value(var)}
    stats = {
        "estado": solver.status_name(estado),
        "objetivo": solver.objective_value / CENT,
        "cota": solver.best_objective_bound / CENT,
        "brecha": _brecha(solver),
        "variables": len(x),
        "variables_afinidad": n_afinidad,
        "unidades_vacias": sum(solver.value(v) for v in vacias),
        "segundos": solver.wall_time,
    }
    return asignacion, stats


def _brecha(solver: cp_model.CpSolver) -> float:
    obj, cota = solver.objective_value, solver.best_objective_bound
    return abs(cota - obj) / max(abs(obj), 1e-9)
