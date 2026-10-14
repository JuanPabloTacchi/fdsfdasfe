import pandas as pd
import pytest

from planogram.afinidad import cargar_afinidad, peso_vecindad


@pytest.fixture
def afinidad(tmp_path):
    pd.DataFrame(
        [("CER", "SNA", 1.0, "manual"), ("LIM", "LAC", -0.8, "manual"), ("CER", "XXX", 0.5, "manual"),
         ("BEB", "SNA", 3.0, "llm")],
        columns=["categoria_a", "categoria_b", "peso", "fuente"],
    ).to_csv(tmp_path / "afinidad.csv", index=False)
    avisos: list[str] = []
    af = cargar_afinidad(tmp_path / "afinidad.csv", {"CER", "SNA", "LIM", "LAC", "BEB"}, avisos)
    return af, avisos


def test_simetrica_y_validada(afinidad):
    af, avisos = afinidad
    assert af["CER", "SNA"] == af["SNA", "CER"] == 1.0
    assert af["BEB", "SNA"] == 1.0  # recortado a [-1, 1]
    assert ("CER", "XXX") not in af
    assert len(avisos) == 2


def test_peso_segun_tipo_de_vecindad(afinidad):
    af, _ = afinidad
    # Enfrentadas en un pasillo cuenta todo; contiguas en la misma cara solo el castigo,
    # para no premiar intercalar categorías tramo por tramo.
    assert peso_vecindad(af, "CER", "SNA", "frente") == 1.0
    assert peso_vecindad(af, "CER", "SNA", "lado") == 0.0
    assert peso_vecindad(af, "LIM", "LAC", "lado") == -0.8
    assert peso_vecindad(af, "LIM", "LIM", "frente") == 0.0


def test_sin_archivo(tmp_path):
    avisos: list[str] = []
    assert cargar_afinidad(tmp_path / "no_existe.csv", {"A"}, avisos) == {}
    assert avisos
