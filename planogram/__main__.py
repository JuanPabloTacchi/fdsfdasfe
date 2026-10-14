"""Línea de comandos.

    python -m planogram generar --datos data/sintetico
    python -m planogram grafo --datos data/sintetico --tienda T01 --salida salida/T01
    python -m planogram optimizar --datos data/sintetico --tienda T01
    python -m planogram blender --tienda T01                 (vista 3D, después de optimizar)
    python -m planogram blender --tienda T01 --version referencia
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from .config import Config
from .ingesta.compatibilidad import resumen_compatibilidad, zona_de
from .ingesta.datos import cargar_categorias, cargar_fichas, cargar_zonas
from .ingesta.grafo import construir_grafo, guardar_grafo, hijos, nodos
from .ingesta.lector_dxf import leer_dxf
from .viz import plot_elevaciones, plot_planta


def main() -> None:
    parser = argparse.ArgumentParser(prog="planogram")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_gen = sub.add_parser("generar", help="genera el conjunto de datos sintético")
    p_gen.add_argument("--datos", type=Path, default=Path("data/sintetico"))
    p_gen.add_argument("--semilla", type=int, default=42)

    p_grafo = sub.add_parser("grafo", help="etapa 1: plano DXF → grafo de la tienda")
    p_grafo.add_argument("--datos", type=Path, default=Path("data/sintetico"))
    p_grafo.add_argument("--tienda", required=True)
    p_grafo.add_argument("--salida", type=Path)

    p_opt = sub.add_parser("optimizar", help="todas las etapas: grafo, scoring, optimización y salidas")
    p_opt.add_argument("--datos", type=Path, default=Path("data/sintetico"))
    p_opt.add_argument("--tienda", required=True)
    p_opt.add_argument("--salida", type=Path)
    p_opt.add_argument("--descomponer", action="store_true",
                       help="mide cuánto aporta cada nivel (macro y micro) a la mejora; agrega una corrida micro")

    p_bl = sub.add_parser("blender", help="abre la tienda optimizada en 3D en Blender")
    p_bl.add_argument("--datos", type=Path, default=Path("data/sintetico"))
    p_bl.add_argument("--tienda", required=True)
    p_bl.add_argument("--salida", type=Path)
    p_bl.add_argument("--camara", help="id de una cara (ej. G-02:E) para verla desde el pasillo")
    p_bl.add_argument("--render", type=Path, help="en vez de abrir Blender, guarda una imagen PNG")
    p_bl.add_argument("--version", choices=["optimizado", "referencia"], default="optimizado",
                      help="qué planograma mostrar: el optimizado o el de la regla de referencia")

    args = parser.parse_args()
    if args.comando == "generar":
        from .sintetico.generar import generar_todo

        generar_todo(args.datos, args.semilla)
        print(f"Datos sintéticos en {args.datos}")
    elif args.comando == "blender":
        abrir_blender(args.datos, args.salida or Path("salida") / args.tienda, args.camara, args.render, args.version)
    elif args.comando == "grafo":
        correr_grafo(args.datos, args.tienda, args.salida or Path("salida") / args.tienda)
    else:
        correr_optimizacion(args.datos, args.tienda, args.salida or Path("salida") / args.tienda, args.descomponer)


def abrir_blender(dir_datos: Path, dir_salida: Path, camara: str | None, render: Path | None,
                  version: str = "optimizado") -> None:
    if not (dir_salida / "planograma.csv").exists():
        raise SystemExit(f"No hay planograma en {dir_salida}: corre primero `python -m planogram optimizar`")
    exe = buscar_blender()
    script = Path(__file__).resolve().parent.parent / "herramientas" / "blender_tienda.py"
    args = [str(dir_salida), "--datos", str(dir_datos), "--version", version]
    if camara:
        args += ["--camara", camara]
    if render:
        subprocess.run([exe, "-b", "--python", str(script), "--", *args, "--render", str(render)], check=True)
        print(f"Imagen en {render}")
    else:
        subprocess.Popen([exe, "--python", str(script), "--", *args])
        print(f"Abriendo Blender ({version})...")


def buscar_blender() -> str:
    """Variable PLANOGRAM_BLENDER, luego el PATH, luego la instalación estándar de Windows."""
    candidatos = [os.environ.get("PLANOGRAM_BLENDER"), shutil.which("blender")]
    candidatos += sorted(str(p) for p in Path("C:/Program Files/Blender Foundation").glob("Blender*/blender.exe"))[::-1]
    for c in candidatos:
        if c and Path(c).exists():
            return c
    raise SystemExit("No encontré Blender. Instálalo o define PLANOGRAM_BLENDER con la ruta a blender.exe")


def correr_optimizacion(dir_datos: Path, tienda: str, dir_salida: Path, descomponer: bool = False) -> None:
    from .pipeline import correr_tienda

    r = correr_tienda(dir_datos, tienda, dir_salida, descomponer=descomponer)
    o, b = r["optimizado"], r["base"]
    print(f"Tienda {tienda}")
    print(f"  {'':24}{'referencia':>14}{'optimizado':>14}")
    for k, nombre in [("utilidad_semanal", "utilidad $/semana"), ("venta_semanal_unidades", "venta unid/semana"),
                      ("skus_listados", "SKUs listados"), ("ocupacion", "ocupación"),
                      ("frentes_promedio", "frentes promedio"), ("recorrido_medio_m", "recorrido medio (m)")]:
        print(f"  {nombre:24}{b[k]:>14,}{o[k]:>14,}")
    print(f"  mejora de utilidad: {r['mejora_utilidad']:+.1%}")
    print("  ciclo categorías ↔ tráfico:")
    for h in r["iteraciones_trafico"]:
        cambio = "" if h["unidades_cambiadas"] is None else f", cambian {h['unidades_cambiadas']:.0%} de las unidades"
        recorrido = "" if h["recorrido_medio_m"] is None else f", recorrido {h['recorrido_medio_m']} m"
        print(f"    iteración {h['iteracion']}: valor estimado ${h['valor_estimado']:,.0f}{recorrido}{cambio}"
              f"{'  ← elegida' if h['elegida'] else ''}")
    for nombre, util in r["descomposicion"].items():
        print(f"    {nombre:22}{util:>14,} ({util / b['utilidad_semanal'] - 1:+.1%} vs referencia)")
    a, ab = r["afinidad"], r["afinidad_base"]
    for k, nombre in [("puntaje", "afinidad (puntaje)"), ("vecindades_afines", "vecindades afines"),
                      ("vecindades_en_conflicto", "vecindades en conflicto"),
                      ("bloques", "bloques de categoría")]:
        print(f"  {nombre:24}{ab[k]:>14,}{a[k]:>14,}")
    print(f"  violaciones: {r['violaciones']} (referencia: {r['violaciones_base']}), avisos del plano: {r['avisos']}")
    m = r["macro"]
    print(f"  macro: {m['estado']}, brecha {m['brecha']:.2%}, {m['variables']} variables, {m['segundos']:.1f} s")
    print(f"  tiempos (s): {r['tiempos_s']}")
    print(f"Salida en {dir_salida}")


def correr_grafo(dir_datos: Path, tienda: str, dir_salida: Path, cfg: Config = Config()) -> None:
    dir_tienda = dir_datos / "tiendas" / tienda
    plano = leer_dxf(dir_tienda / "plano.dxf", cfg)
    avisos: list[str] = []
    fichas = cargar_fichas(dir_tienda / "gondolas.csv")
    zonas = cargar_zonas(dir_tienda / "zonas.csv", plano.zonas, avisos)
    categorias, reglas = cargar_categorias(dir_datos)
    G, avisos_grafo = construir_grafo(tienda, plano, fichas, zonas, categorias, reglas, cfg)
    avisos += avisos_grafo

    dir_salida.mkdir(parents=True, exist_ok=True)
    guardar_grafo(G, dir_salida / "grafo.json")
    compat = resumen_compatibilidad(G)
    compat.to_csv(dir_salida / "compatibilidad.csv", index=False)
    (dir_salida / "avisos.txt").write_text("\n".join(avisos), encoding="utf-8")
    plot_planta(G, dir_salida / "planta.png")
    plot_elevaciones(G, caras_de_muestra(G), dir_salida / "elevaciones.png")

    conteo = {}
    for _, d in G.nodes(data=True):
        conteo[d["nodo"]] = conteo.get(d["nodo"], 0) + 1
    print(f"Tienda {tienda}: " + ", ".join(f"{k}={v}" for k, v in conteo.items()))
    print(f"Aristas: {G.number_of_edges()}")
    print(f"\nCompatibilidad categoría → espacio disponible:\n{compat.to_string(index=False)}")
    print(f"\nAvisos ({len(avisos)}):" + "".join(f"\n  - {a}" for a in avisos))
    print(f"\nSalida en {dir_salida}")


def caras_de_muestra(G) -> list[str]:
    """Una cara por cada tipo de zona y una sin zona, para revisar la contención."""
    elegidas: dict[str | None, str] = {}
    for c, _ in nodos(G, "cara"):
        tipos = set()
        for t in hijos(G, c, "TIENE_TRAMO"):
            for r in hijos(G, t, "TIENE_REPISA"):
                z = zona_de(G, r)
                tipos.add(G.nodes[z]["tipo_zona"] if z else None)
        clave = max((t for t in tipos if t), default=None)
        elegidas.setdefault(clave, c)
    return [elegidas[k] for k in sorted(elegidas, key=lambda k: (k is None, k or ""))]


if __name__ == "__main__":
    main()
