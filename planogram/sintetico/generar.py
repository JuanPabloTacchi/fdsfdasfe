"""Genera el conjunto de datos sintético completo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .afinidad import generar_afinidad
from .catalogo import TIENDAS, generar_catalogo
from .plano import ParamsPlano, generar_plano


def generar_todo(dir_salida: Path, semilla: int = 42) -> None:
    rng = np.random.default_rng(semilla)
    generar_catalogo(dir_salida, rng)
    generar_afinidad(dir_salida)
    for tienda_id, _, _, _, n_pasillos, largo in TIENDAS:
        generar_plano(dir_salida / "tiendas" / tienda_id, ParamsPlano(n_pasillos=n_pasillos, largo_gondola=largo))
