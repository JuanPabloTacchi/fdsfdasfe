"""Parámetros de la ingesta. Todas las distancias están en metros."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capas:
    """Nombres de capa del DXF (se comparan sin distinguir mayúsculas)."""

    contorno: str = "CONTORNO"
    muro: str = "MURO"
    gondola: str = "GONDOLA"
    zona: str = "ZONA"
    entrada: str = "ENTRADA"
    caja: str = "CAJA"


@dataclass(frozen=True)
class Config:
    capas: Capas = field(default_factory=Capas)

    # Si el DXF no declara $INSUNITS se asume que está en milímetros.
    escala_por_defecto: float = 0.001
    # Muros dibujados como líneas se engrosan a este espesor.
    espesor_muro: float = 0.2

    # Una cara de góndola es válida si mide al menos esto...
    largo_min_cara: float = 0.6
    # ...y si esta franja frente a ella es caminable en al menos esta fracción.
    prof_frente: float = 0.6
    frac_frente_caminable: float = 0.6

    # Navmesh: lado de cada celda y fracción caminable mínima para conservarla.
    tam_celda: float = 0.5
    frac_celda_caminable: float = 0.5
    # Celdas a esta distancia de una cara (o de un punto de interés) quedan conectadas a ella.
    alcance_cara_pasillo: float = 0.9

    # Una repisa pertenece a una zona si al menos esta fracción de su volumen cae dentro.
    umbral_contencion: float = 0.5
    # Fracciones entre esta tolerancia y 1 - tolerancia se reportan como contención parcial.
    tolerancia_parcial: float = 0.05


@dataclass(frozen=True)
class Modelo:
    """Parámetros del modelo de negocio y de las etapas 2 a 4."""

    # Demanda: venta = base × (frentes / frentes_ref)^beta × atractivo^gamma
    beta_espacio: float = 0.17  # elasticidad al espacio (media del meta-análisis de Eisend, 2014)
    gamma_posicion: float = 0.5  # amortigua el efecto de la ubicación sobre la demanda
    reposiciones_semana: float = 3.0  # cuántas veces por semana se repone una repisa
    max_frentes: int = 8

    # Etapa 2: tráfico simulado.
    n_compradores: int = 400
    paradas_por_compra: int = 6  # tráfico de referencia: tramos visitados al azar por cliente
    paradas_aleatorias: int = 1  # tráfico por listas: paradas sin planificar además de la lista
    piso_trafico: float = 0.2  # tráfico normalizado mínimo, para que ninguna repisa valga 0
    peso_cabecera: float = 1.3
    peso_isla: float = 1.1
    bonus_cercania_entrada: float = 0.2
    altura_ojos: float = 1.4  # m, centro de la repisa con mejor visibilidad
    ancho_vertical: float = 0.45  # m, cuánto cae la visibilidad al alejarse de esa altura
    piso_vertical: float = 0.45  # visibilidad mínima (repisa a ras de piso)

    # Etapa 3a: bonificación por tramos contiguos de la misma categoría (fracción del valor de un tramo).
    bonus_contiguidad: float = 0.10
    # Afinidad entre categorías vecinas: peso del par × esto × utilidad de un tramo promedio.
    peso_afinidad: float = 0.3
    ancho_max_pasillo: float = 4.0  # m, tramos enfrentados a más distancia no se consideran vecinos
    # Costo por cada categoría distinta en una misma cara (fracción del valor de un tramo promedio).
    # Evita caras "de sobras" con muchas categorías en pedazos chicos.
    costo_categoria_por_cara: float = 0.25
    # Costo de dejar una unidad sin categoría (fracción del valor de un tramo promedio).
    costo_unidad_vacia: float = 3.0
    tiempo_macro: float = 30.0  # s por iteración

    # Ciclo layout ↔ tráfico: se reoptimiza con el tráfico que genera cada layout.
    iteraciones_trafico: int = 4  # 0 = tráfico fijo (aleatorio), sin ciclo
    mezcla_trafico: float = 0.5  # peso del tráfico nuevo al mezclarlo con el anterior (amortigua oscilaciones)
    tolerancia_cambio: float = 0.02  # se corta si cambia menos de esta fracción de las unidades

    # Etapa 3b: fracción del espacio de la categoría que se planifica (el resto absorbe la fragmentación).
    holgura_micro: float = 0.95
    candidatas_por_sku: int | None = None  # None = todas las repisas donde cabe; acotar solo con catálogos enormes
    tiempo_micro: float = 10.0  # s por categoría

    semilla: int = 0
