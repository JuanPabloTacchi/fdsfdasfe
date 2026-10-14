"""Genera categorías, reglas de zona, SKUs, tiendas y ventas sintéticas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..demanda import PARAMS_POR_ROL


@dataclass(frozen=True)
class DefCategoria:
    id: str
    nombre: str
    rol: str  # destino | rutina | impulso | conveniencia
    n_skus: int
    ancho: tuple[float, float]  # rango en metros
    alto: tuple[float, float]
    prof: tuple[float, float]
    precio: tuple[float, float]
    margen: tuple[float, float]  # fracción del precio
    demanda: float  # unidades semanales medias por SKU
    marcas: tuple[str, ...]
    # (tipo_zona, obligatoria): si es obligatoria la categoría solo puede ir en ese tipo de zona.
    zonas: tuple[tuple[str, bool], ...] = ()


CATEGORIAS = [
    DefCategoria("LAC", "Lácteos", "destino", 200, (0.06, 0.10), (0.08, 0.22), (0.06, 0.10), (0.8, 4.0), (0.18, 0.28), 30,
                 ("Colún", "Soprole", "Nestlé", "Loncoleche"), (("REFRIGERADO", True),)),
    DefCategoria("FIA", "Fiambres y quesos", "rutina", 125, (0.12, 0.20), (0.03, 0.08), (0.15, 0.22), (2.0, 9.0), (0.25, 0.35), 12,
                 ("San Jorge", "PF", "Llanquihue"), (("REFRIGERADO", True),)),
    DefCategoria("BEB", "Bebidas", "rutina", 175, (0.08, 0.12), (0.22, 0.34), (0.08, 0.12), (0.9, 3.0), (0.20, 0.30), 25,
                 ("Coca-Cola", "CCU", "Andina"), (("REFRIGERADO", False),)),
    DefCategoria("CER", "Cervezas", "impulso", 100, (0.07, 0.10), (0.15, 0.25), (0.07, 0.10), (1.0, 3.5), (0.22, 0.32), 18,
                 ("Cristal", "Escudo", "Kunstmann"), (("REFRIGERADO", False),)),
    DefCategoria("CON", "Congelados", "rutina", 125, (0.15, 0.25), (0.04, 0.10), (0.15, 0.25), (2.0, 8.0), (0.22, 0.32), 10,
                 ("Minuto Verde", "Frutos del Maipo", "Super Pollo"), (("CONGELADO", True),)),
    DefCategoria("HEL", "Helados", "impulso", 75, (0.10, 0.20), (0.08, 0.14), (0.10, 0.20), (2.5, 7.0), (0.30, 0.40), 8,
                 ("Savory", "Bresler", "Trendy"), (("CONGELADO", True),)),
    DefCategoria("LIC", "Licores", "conveniencia", 60, (0.08, 0.11), (0.28, 0.34), (0.08, 0.11), (8.0, 30.0), (0.25, 0.35), 3,
                 ("Mistral", "Capel", "Johnnie Walker"), (("SEGURIDAD", True),)),
    DefCategoria("SNA", "Snacks", "impulso", 175, (0.15, 0.25), (0.20, 0.30), (0.06, 0.10), (0.8, 3.0), (0.30, 0.42), 15,
                 ("Lay's", "Marco Polo", "Evercrisp")),
    DefCategoria("GAL", "Galletas y dulces", "impulso", 175, (0.05, 0.20), (0.05, 0.20), (0.05, 0.10), (0.5, 2.5), (0.30, 0.40), 14,
                 ("Costa", "McKay", "Arcor")),
    DefCategoria("ABA", "Abarrotes", "destino", 200, (0.08, 0.15), (0.15, 0.30), (0.05, 0.10), (0.8, 3.5), (0.12, 0.20), 22,
                 ("Carozzi", "Lucchetti", "Tucapel", "Miraflores")),
    DefCategoria("CNS", "Conservas", "rutina", 125, (0.07, 0.10), (0.05, 0.12), (0.07, 0.10), (0.9, 3.0), (0.20, 0.30), 9,
                 ("Robinson Crusoe", "Wasil", "San Remo")),
    DefCategoria("DES", "Desayuno", "rutina", 150, (0.10, 0.20), (0.15, 0.30), (0.05, 0.10), (1.5, 6.0), (0.22, 0.32), 10,
                 ("Nestlé", "Kellogg's", "Nescafé")),
    DefCategoria("LIM", "Limpieza", "destino", 175, (0.10, 0.20), (0.20, 0.35), (0.07, 0.12), (1.0, 6.0), (0.20, 0.30), 9,
                 ("Clorox", "Omo", "Quix", "Virginia")),
    DefCategoria("CPE", "Cuidado personal", "conveniencia", 150, (0.05, 0.10), (0.12, 0.25), (0.04, 0.08), (1.5, 8.0), (0.30, 0.45), 6,
                 ("Dove", "Head & Shoulders", "Colgate")),
    DefCategoria("MAS", "Mascotas", "conveniencia", 75, (0.20, 0.35), (0.20, 0.36), (0.08, 0.15), (3.0, 15.0), (0.20, 0.30), 4,
                 ("Champion", "Dog Chow", "Whiskas")),
]

TIENDAS = [
    # tienda_id, nombre, formato, factor_demanda, n_pasillos, largo_gondola
    ("T01", "Tienda Centro", "supermercado", 1.0, 6, 12.0),
    ("T02", "Tienda Barrio", "express", 0.6, 5, 10.0),
    ("T03", "Tienda Mall", "hiper", 1.6, 8, 14.0),
]


def generar_catalogo(dir_salida: Path, rng: np.random.Generator, semanas: int = 26) -> None:
    dir_salida.mkdir(parents=True, exist_ok=True)

    # Probabilidad de estar en la lista de compras y fracción de impulso: valor del rol ± 20 %.
    filas = []
    for c in CATEGORIAS:
        p, a = PARAMS_POR_ROL[c.rol]
        filas.append((c.id, c.nombre, c.rol, round(p * rng.uniform(0.8, 1.2), 3), round(a * rng.uniform(0.8, 1.2), 3)))
    pd.DataFrame(filas, columns=["categoria_id", "nombre", "rol", "prob_lista", "frac_impulso"]
                 ).to_csv(dir_salida / "categorias.csv", index=False)

    pd.DataFrame(
        [(c.id, tipo, int(oblig)) for c in CATEGORIAS for tipo, oblig in c.zonas],
        columns=["categoria_id", "tipo_zona", "obligatoria"],
    ).to_csv(dir_salida / "categoria_zona.csv", index=False)

    pd.DataFrame(
        [(t[0], t[1], t[2], t[3]) for t in TIENDAS], columns=["tienda_id", "nombre", "formato", "factor_demanda"]
    ).to_csv(dir_salida / "tiendas.csv", index=False)

    skus = []
    for c in CATEGORIAS:
        for k in range(c.n_skus):
            # Una de cada ~6 es marca propia, con margen más alto.
            propia = rng.random() < 0.17
            marca = "Marca Propia" if propia else rng.choice(c.marcas)
            precio = rng.uniform(*c.precio) * (0.8 if propia else 1.0)
            margen = rng.uniform(*c.margen) + (0.08 if propia else 0.0)
            skus.append({
                "sku_id": f"{c.id}{k + 1:03d}",
                "categoria_id": c.id,
                "marca": marca,
                "descripcion": f"{c.nombre} {marca} #{k + 1}",
                "ancho": round(rng.uniform(*c.ancho), 3),
                "alto": round(rng.uniform(*c.alto), 3),
                "prof": round(rng.uniform(*c.prof), 3),
                "precio": round(precio, 2),
                "costo": round(precio * (1 - margen), 2),
                # Demanda base con cola larga: pocos SKUs venden mucho.
                "_demanda": c.demanda * rng.lognormal(mean=-0.5, sigma=1.0),
            })
    df_skus = pd.DataFrame(skus)
    df_skus.drop(columns="_demanda").to_csv(dir_salida / "skus.csv", index=False)

    filas = []
    for tienda_id, _, _, factor, *_ in TIENDAS:
        # Cada tienda tiene preferencias propias por SKU.
        gusto = rng.lognormal(mean=0.0, sigma=0.3, size=len(df_skus))
        lam_base = df_skus["_demanda"].to_numpy() * factor * gusto
        for semana in range(1, semanas + 1):
            estacion = 1 + 0.15 * np.sin(2 * np.pi * semana / 52)
            unidades = rng.poisson(lam_base * estacion)
            for sku_id, u in zip(df_skus["sku_id"], unidades):
                filas.append((semana, tienda_id, sku_id, int(u)))
    pd.DataFrame(filas, columns=["semana", "tienda_id", "sku_id", "unidades"]).to_csv(
        dir_salida / "ventas.csv", index=False
    )
