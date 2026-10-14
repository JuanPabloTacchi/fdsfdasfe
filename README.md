# Planogram

Genera planogramas de tienda a partir de un plano CAD. El diseño está en
[docs/arquitectura_layout.drawio](docs/arquitectura_layout.drawio).

Estado: **prototipo completo** (las 4 etapas), probado con datos sintéticos.

| Etapa | Qué hace | Código |
|---|---|---|
| 1. Grafo | DXF → góndolas, caras, tramos, repisas, zonas y navmesh de pasillos | `planogram/ingesta/` |
| 2. Scoring | tráfico simulado × visibilidad × altura → score y multiplicador de demanda de cada repisa | `planogram/scoring.py` |
| 3a. Macro | qué categoría va en cada tramo (CP-SAT) | `planogram/optim/macro.py` |
| 3b. Micro | qué SKUs, con cuántos frentes y en qué repisa (análisis marginal + CP-SAT) | `planogram/optim/micro.py` |
| 4. Salida | planograma CSV/PDF/DXF, mapas, KPIs y comparación con una regla de referencia | `planogram/salida.py`, `planogram/evaluacion.py` |

## Uso

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]

.venv/Scripts/python -m planogram generar                 # datos sintéticos en data/sintetico
.venv/Scripts/python -m planogram grafo --tienda T01      # solo etapa 1, para revisar el plano
.venv/Scripts/python -m planogram optimizar --tienda T01  # todas las etapas (~3-4 min)
.venv/Scripts/python -m planogram optimizar --tienda T01 --descomponer  # + aporte de macro y micro
.venv/Scripts/python -m planogram blender --tienda T01    # vista 3D en Blender (4.x)
.venv/Scripts/python -m planogram blender --tienda T01 --camara G-02:E   # desde un pasillo
.venv/Scripts/python -m planogram blender --tienda T01 --version referencia  # la regla simple, para comparar
.venv/Scripts/python -m pytest
```

Salidas en `salida/<tienda>/`:

| Archivo | Contenido |
|---|---|
| `planograma.csv` | una fila por SKU ubicado: góndola, cara, tramo, nivel, x, frentes, venta y utilidad esperadas |
| `planogramas.pdf`, `planograma_muestra.png` | vista frontal de cada cara con cada frente dibujado |
| `planograma.dxf` | el plano original con una capa `PLANOGRAMA` que rotula la categoría de cada tramo |
| `categorias_optimizado.png`, `categorias_base.png` | categoría de cada tramo, optimizado vs. referencia |
| `trafico.png`, `trafico_base.png` | tráfico que genera cada layout (misma escala) y score de cada tramo |
| `iteraciones_trafico.csv` | cada vuelta del ciclo categorías ↔ tráfico: valor, recorrido medio, cuánto cambió |
| `pasillos.csv` | qué categorías ve el cliente en cada pasillo (las dos caras enfrentadas) |
| `conflictos_afinidad.csv` | categorías que quedaron vecinas y deberían estar separadas |
| `categorias.csv` | metros, SKUs y utilidad por categoría, optimizado vs. referencia |
| `resumen.json` | KPIs, estado de los solvers y tiempos |
| `violaciones.txt` | chequeo independiente de todas las restricciones (debe decir "sin violaciones") |
| `avisos.txt` | problemas detectados en el plano |

## Modelo

Utilidad semanal de un SKU con *f* frentes en una repisa (`planogram/demanda.py`):

```
venta     = demanda_base × multiplicador_repisa × (f / f_ref)^β
capacidad = f × unidades_al_fondo × reposiciones_por_semana
utilidad  = margen × min(venta, capacidad)
```

- `demanda_base`: promedio semanal histórico; se asume observado con `f_ref` frentes en una ubicación promedio.
- `f_ref`: frentes para no quebrar stock entre reposiciones.
- β = 0,17: elasticidad al espacio (media del meta-análisis de Eisend, 2014).
- `multiplicador_repisa` = ((1 − impulso) + impulso × exposición) × visibilidad^γ, con γ = 0,5:
  la venta planificada no depende de cuánta gente pasa; la de impulso, sí. `exposición` son
  las pasadas frente a la repisa relativas a un tráfico de referencia fijo, así que un layout
  que hace caminar más vende más por impulso en total.

### Tráfico ↔ categorías

`categorias.csv` trae por categoría `prob_lista` (probabilidad de que el cliente venga a
buscarla) y `frac_impulso` (fracción de su venta que depende de pasar frente a ella). Hoy
son sintéticos según el rol; con boletas se estiman.

La etapa 3a itera (`planogram/optim/ciclo.py`): optimiza categorías con el tráfico actual,
simula el tráfico que genera ese layout (cada cliente busca los productos de su lista donde
quedaron y sale por la caja), mezcla ese tráfico con el anterior y repite. Se queda con la
iteración de mayor valor evaluada con su propio tráfico. La etapa 3b (SKUs) usa el tráfico
del layout elegido, y la referencia se evalúa con el tráfico que genera su propio layout.
`iteraciones_trafico.csv` muestra cada vuelta; `trafico.png` y `trafico_base.png`, el
tráfico de cada layout con la misma escala.

Todos los parámetros están en `planogram/config.py` (`Modelo`). **Son supuestos,
no estimaciones**: con datos reales hay que calibrar β, γ y el scoring.

### Afinidad entre categorías

`afinidad_categorias.csv` dice qué categorías conviene tener cerca (peso > 0) y
cuáles separar (peso < 0). Hoy es sintético (`planogram/sintetico/afinidad.py`);
la idea es reemplazarlo por un paso que lo infiera (LLM, embeddings o co-compra
en boletas) y escriba el mismo archivo. La columna `fuente` registra el origen.

Cómo cuenta en el nivel macro (`planogram/afinidad.py`, `peso_vecindad`):

- tramos **enfrentados en un pasillo**: cuenta el peso completo;
- tramos **contiguos en la misma cara**: solo cuenta el castigo (peso < 0). Premiar
  la afinidad ahí hace que convenga intercalar categorías tramo por tramo.

Además, el nivel macro paga un costo por cada categoría distinta en una misma
cara (evita caras "de sobras") y uno alto por dejar unidades vacías.

**Referencia** (`planogram/optim/base.py`): la práctica habitual sin optimización.
Espacio por categoría proporcional a su utilidad, en orden de lectura, y frentes
proporcionales a la venta, llenando cada categoría.

### Vista 3D en Blender

`herramientas/blender_tienda.py` arma la escena con el Python de Blender (no necesita
el entorno del proyecto) a partir de `grafo.json` y `planograma.csv`: piso y muros,
góndolas con repisas, una caja por cada frente de producto con el color de su
categoría (los mismos colores que los mapas 2D), zonas semitransparentes y el
nombre de la categoría sobre cada bloque. Las colecciones Estructura, Productos
(un objeto por categoría), Zonas, Etiquetas y Tráfico se prenden y apagan en el
panel de la derecha; Tráfico arranca apagada. Con `--render` guarda una imagen sin
abrir la interfaz. Si Blender no está en la ruta estándar, define `PLANOGRAM_BLENDER`.

## Datos de entrada

```
data/<conjunto>/
  categorias.csv        categoria_id, nombre, rol
  categoria_zona.csv    categoria_id, tipo_zona, obligatoria (0/1)
  afinidad_categorias.csv  categoria_a, categoria_b, peso (-1..1), fuente
  skus.csv              sku_id, categoria_id, marca, descripcion, ancho, alto, prof, precio, costo
  tiendas.csv           tienda_id, nombre, formato, factor_demanda
  ventas.csv            semana, tienda_id, sku_id, unidades
  tiendas/<tienda_id>/
    plano.dxf
    gondolas.csv        gondola_id, tipo, alto, niveles ("0.15;0.55;..."), ancho_tramo,
                        prof_max_repisa, espesor_repisa, caras_cortas (0/1)
    zonas.csv           zona_id, tipo, z_min, z_max, exclusiva (0/1)
```

Medidas en metros. El DXF puede venir en cualquier unidad declarada en `$INSUNITS`.

### Plano DXF

Polilíneas cerradas por capa, con el id escrito como texto (TEXT/MTEXT) de la
misma capa dentro del polígono:

| Capa | Contenido |
|---|---|
| `CONTORNO` | perímetro de la sala de ventas |
| `MURO` | líneas o polígonos (las líneas se engrosan a 0,2 m) |
| `GONDOLA` | un rectángulo por góndola; el id enlaza con `gondolas.csv` |
| `ZONA` | planta de cada zona; el id enlaza con `zonas.csv` (que trae su rango de altura) |
| `ENTRADA`, `CAJA` | puntos de interés para el tráfico; las cajas son obstáculos |

Los nombres de capa se configuran en `planogram/config.py`.

### Reglas de zona

Una repisa pertenece a una zona si al menos el 50 % de su volumen libre cae
dentro del cubo (polígono × [z_min, z_max]). Luego:

- repisa sin zona → acepta categorías sin zona obligatoria;
- repisa en zona de tipo T → acepta las categorías permitidas en T y, si la
  zona no es exclusiva, también las que no tienen zona obligatoria.
