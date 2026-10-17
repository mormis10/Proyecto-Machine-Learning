# Proyecto-Machine-Learning
Proyecto del Curso Machine Learning UCR (CI-0148): **Premier League Match Predictor**.
Predice el resultado de un partido (`FTR`: H / D / A) con Logistic Regression y Gradient Boosting,
usando solo información disponible antes del partido.

## Instalando el ambiente y dependencias para correr el sistema

```
python -m venv .venv
.venv\Scripts\activate          # en Windows
# source .venv/bin/activate     # en Mac/Linux
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `matplotlib` y `pytest`.

## Regenerar los datos y correr los tests

Los parquet **no están en git**: se generan desde `data/raw/`. Después de clonar, desde la raíz del repo:

```
python src/concat_seasons.py    # data/raw/*.csv -> data/interim/matches_clean.parquet
pytest -v                       # todos los tests deben pasar
python src/features.py          # matches_clean.parquet -> data/processed/features.parquet
python src/quality_report.py    # -> docs/quality_report.txt
python src/eda.py               # -> docs/figures/*.png
```

`src/make_manifest.py` solo hay que correrlo si cambian los CSV de `data/raw/`.


## Estructura

```
data/raw/          33 CSV de DataHub (93/94-25/26) + MANIFEST.csv. No se editan nunca.
data/interim/      matches_clean.parquet (generado, ignorado por git)
data/processed/    features.parquet (generado, ignorado por git)
docs/data_card.md  fuente, decisiones y su justificación
docs/fase2_preparacion_datos.tex / .pdf   informe de la Fase 2
docs/quality_report.txt   faltantes y clases (generado)
docs/figures/      figuras del EDA (generadas por src/eda.py, usadas por el informe)
src/
  concat_seasons.py   une las temporadas y limpia -> matches_clean.parquet
  team_names.py       diccionario de normalización de nombres
  features.py         todas las features y build_features (ver abajo)
  splits.py           particiones walk-forward
  models.py           pipelines de Logistic Regression y Gradient Boosting
  quality_report.py   reporte de faltantes y distribución de clases
  eda.py              figuras del análisis exploratorio
  make_manifest.py    genera data/raw/MANIFEST.csv
  names_conventions_detections.py, takes_coutes.py   scripts de exploración, fuera del pipeline
tests/             un archivo por paso del pipeline
```

## `matches_clean.parquet`

12,704 partidos (462 en 93/94 y 94/95, 380 en el resto; 25/26 está completa), 24 columnas.

- `Date` ya es datetime. El parseo acepta ISO, `dd/mm/yyyy` y `dd/mm/yy`, y falla si aparece otro formato.
- Nombres de equipo normalizados con `team_names.py` (51 equipos, nombres canónicos = los de DataHub).
- Ordenado por `["Date", "HomeTeam"]` con índice 0..12703. **Ese índice es el `match_id`** que usan todas las features para volver a unirse.

## Features (`src/features.py`)

| Función | Entrada → salida | Qué hace |
|---|---|---|
| `to_long(m)` | 1 fila por partido → 2 filas (una por equipo) | `team`, `opponent`, `is_home`, `gf`, `ga`, tiros, `points` desde la perspectiva de cada equipo |
| `compute_elo(m)` | 1 fila por partido | `elo_home`, `elo_away`, `elo_diff` **antes** del partido |
| `add_form(long)` | formato largo | `form_points/gf/ga` (últimos 5) y `form_venue_*` (últimos 5 en la misma condición) |
| `add_shots(long)` | formato largo | `roll_shots_for/against`, `roll_sot_for/against` (últimos 5) |
| `add_flag(long)` | formato largo | `pocos_partidos` = 1 si el equipo aún no tiene 5 partidos previos (de resultados o de tiros) |
| `fill_prior(long)` | formato largo | rellena los `NaN` con la media de la liga hasta la fecha anterior (por sede en `form_venue_*`) |
| `to_wide(long, cols)` | formato largo → 1 fila por partido | columnas `home_*` y `away_*`, índice = `match_id` |
| `add_diffs(wide)` | 1 fila por partido | `diff_* = home_* - away_*` para las 10 features de `TEAM_COLS` |
| `build_features(m)` | `matches_clean` → `features.parquet` | encadena todo, une Elo y el target `FTR`, filtra el burn-in **al final** y descarta la primera jornada sin prior |

Parámetros del Elo: 1500 inicial, K = 20, +60 de localía, 80% de arrastre entre temporadas,
ascendidos con el promedio de los descendidos. La forma y los tiros cruzan temporadas (la ventana no se reinicia en agosto).

**Resultado:** `features.parquet` tiene 9,871 partidos (2000/01-2025/26) y 39 columnas. Desde 2000/01 hay 9,880 partidos;
se descartan 9, los de la primera jornada de 2000/01, que no tienen ningún partido previo con tiros ni para calcular un prior.

**`MODEL_COLS`** son las columnas que reciben **ambos** modelos: `elo_diff`, las 10 `diff_*` y las banderas
`home_pocos_partidos` y `away_pocos_partidos`. No se meten `home_x` ni `away_x` junto con `diff_x`
(colinealidad perfecta para Logistic Regression).

### Reglas contra fuga de información

1. **Toda feature usa solo partidos anteriores.** Cualquier ventana móvil pasa por `rolling_prev`, que hace `shift(1)` antes del `rolling`. El valor previo (`fill_prior`) usa solo fechas estrictamente anteriores.
2. **Se calcula sobre todas las temporadas y se filtra al final.** 93/94-99/00 son burn-in (`TRAIN_START = 2000`): alimentan el Elo y la forma pero no entran a entrenamiento. Filtrar antes de calcular rompe el Elo sin dar error (`test_burn_in_aplicado` lo detecta).
3. **Cada feature nueva lleva tests de recálculo y de perturbación**, como en `test_form.py` y `test_tiros.py`: recalcular a mano filas al azar con datos previos, y alterar un partido y comprobar que nada anterior cambia.
4. Columnas del mismo partido (tiros, corners, tarjetas, medio tiempo) nunca como feature directa. Ver `docs/data_card.md`.
5. **El relleno de faltantes va después del `rolling`**, nunca antes, y siempre acompañado de la bandera `pocos_partidos`.

## Particiones y modelos

- `splits.py`: `walk_forward_splits(feats)` genera un `Fold` por temporada de prueba (2020/21 a 2025/26), entrenando con todo lo anterior.
  Dentro del entrenamiento, la última temporada es la validación interna para hiperparámetros; nunca se usa la de prueba.
  Cada fold verifica con un `assert` que `max(fecha_train) < min(fecha_test)`.
- 2020/21 (sin público) se marca como atípica (`test_atypical`, `val_atypical`), pero no se excluye. Al probar en 2021/22, la validación interna cae en 2020/21.
- `models.py`: `get_xy` devuelve siempre `MODEL_COLS`. `make_logreg()` lleva `StandardScaler` dentro del `Pipeline` (se ajusta solo con el entrenamiento de cada fold);
  `make_gboost()` no escala.

## Estado de la Fase 2

### Completado
- **Ingesta:** estructura del repo, CSV crudos, `MANIFEST.csv`, `requirements.txt`.
- **Limpieza:** concatenación, parseo de fechas, nombres, orden cronológico, validaciones (tests) y `matches_clean.parquet`.
- **Columnas permitidas:** lista de prohibidas, cuotas (el mirror no las trae) y `Referee` excluida. Anotado en el data card.
- **Features:** formato largo, Elo, burn-in, forma reciente, promedios de tiros, diferencias local menos visitante, bandera `pocos_partidos`, valor previo y `build_features` + `features.parquet`.
- **Particiones:** walk-forward 2020/21-2025/26, validación interna y 2020/21 marcada como atípica.
- **Preprocesamiento:** pipelines con las mismas columnas para ambos modelos.
- **Calidad:** tests de recálculo y perturbación (Elo, forma, tiros, bandera y valor previo), assert por fold, reporte de calidad.
- **EDA:** H/D/A por temporada, victoria según Elo por localía, matriz de correlación.
- **Documentos:** data card, README e informe de la fase (`docs/fase2_preparacion_datos.pdf`).

### Pendiente

- **Corregir el informe de la Fase 1 destacando secciones como revisión literaria y referencias**