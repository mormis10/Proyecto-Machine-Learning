# Proyecto-Machine-Learning
Proyecto del Curso Machine Learning UCR (CI-0148): **Premier League Match Predictor**.
Predice el resultado de un partido (`FTR`: H / D / A) con Logistic Regression y Gradient Boosting,
usando solo información disponible antes del partido.

## Installing the Environment and Dependencies to Run the Project

```
python -m venv .venv
.venv\Scripts\activate          # en Windows
# source .venv/bin/activate     # en Mac/Linux
pip install -r requirements.txt
```

## Regenerar los datos y correr los tests

Los parquet **no están en git**: se generan desde `data/raw/`. Después de clonar, desde la raíz del repo:

```
python src/concat_seasons.py    # data/raw/*.csv -> data/interim/matches_clean.parquet
pytest -v                       # todos los tests deben pasar
```

`pytest.ini` agrega `src/` al path, así que los tests importan directo (`from features import ...`).
`tests/conftest.py` define la fixture `m` (el parquet limpio) disponible en todos los tests.

## Estructura

```
data/raw/          33 CSV de DataHub (93/94-25/26) + MANIFEST.csv. No se editan nunca.
data/interim/      matches_clean.parquet (generado, ignorado por git)
data/processed/    features.parquet (pendiente)
docs/data_card.md  fuente, decisiones y su justificación
src/
  concat_seasons.py   une las temporadas y limpia -> matches_clean.parquet
  team_names.py       diccionario de normalización de nombres
  features.py         todas las features (ver abajo)
  make_manifest.py    genera data/raw/MANIFEST.csv
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
| `to_wide(long, TEAM_COLS)` | formato largo → 1 fila por partido | columnas `home_*` y `away_*`, índice = `match_id` |
| `add_diffs(wide)` | 1 fila por partido | `diff_* = home_* - away_*` para las 10 features de `TEAM_COLS` |

Parámetros del Elo: 1500 inicial, K = 20, +60 de localía, 80% de arrastre entre temporadas,
ascendidos con el promedio de los descendidos. La forma y los tiros cruzan temporadas (la ventana no se reinicia en agosto).

### Reglas contra fuga de información

1. **Toda feature usa solo partidos anteriores.** Cualquier ventana móvil pasa por `rolling_prev`, que hace `shift(1)` antes del `rolling`.
2. **Se calcula sobre todas las temporadas y se filtra al final.** 93/94-99/00 son burn-in (`TRAIN_START = 2000`): alimentan el Elo y la forma pero no entran a entrenamiento. Filtrar antes de calcular rompe el Elo sin dar error (`test_burn_in_aplicado` lo detecta).
3. **Cada feature nueva lleva tests de recálculo y de perturbación**, como en `test_form.py` y `test_tiros.py`: recalcular a mano filas al azar con datos previos, y alterar un partido y comprobar que nada anterior cambia.
4. Columnas del mismo partido (tiros, corners, tarjetas, medio tiempo) nunca como feature directa. Ver `docs/data_card.md`.

## Estado de la Fase 2

### Hecho
- **Ingesta:** estructura del repo, CSV crudos, `MANIFEST.csv`, `requirements.txt`.
- **Limpieza:** concatenación, parseo de fechas, nombres, orden cronológico, validaciones (tests) y `matches_clean.parquet`.
- **Columnas permitidas:** lista de prohibidas, cuotas (el mirror no las trae) y `Referee` excluida. Anotado en el data card.
- **Features:** formato largo, Elo, burn-in, forma reciente, promedios de tiros y diferencias local menos visitante.

### Pendiente

**Ingeniería de características**
- **Bandera `pocos_partidos` y valor previo para ascendidos.** Hoy quedan `NaN` en el primer partido de cada equipo: 51 en `form_*` (102 en `form_venue_*`) y, desde 2000/01, 46 en `roll_*` (la primera jornada de 2000/01 y los equipos sin partidos previos con estadísticas, como Brentford 2021 o Luton 2023). Logistic Regression no acepta `NaN`.
- **`build_features(m)` + `data/processed/features.parquet`.** Unir `m` (target `FTR`, `Season`, `Date`) + `compute_elo` + `add_diffs(to_wide(...))` por `match_id`. El filtro `SeasonStart >= TRAIN_START` va **al final**, y el `to_parquet` en el `if __name__ == "__main__"`, para que los tests puedan llamar a la función sin escribir archivos.
- **Revisión cruzada** del código de features (quien no lo escribió).

**Rango y particiones walk-forward**
- Filtro desde 2000/01: usar `TRAIN_START`.
- Folds: probar en 2020/21 a 2024/25, entrenando con todo lo anterior. 25/26 está completa (380 partidos), así que se puede agregar.
- Validación interna para hiperparámetros: la última temporada del bloque de entrenamiento, nunca la de prueba.
- Marcar 2020/21 como temporada atípica (sin público).

**Preprocesamiento por modelo**
- Pipeline de scikit-learn con `StandardScaler` solo para Logistic Regression, ajustado dentro de cada fold.
- Mismas columnas para ambos modelos. Ojo: `diff_x` es exactamente `home_x - away_x`, así que meter las tres es colinealidad perfecta para Logistic Regression. Propuesta: usar diferencias + `elo_diff`.

**Aseguramiento de calidad**
- Ya existen tests de recálculo y perturbación para Elo, forma y tiros.
- Falta: assert en cada fold de `max(fecha_train) < min(fecha_test)`, y reporte de calidad (faltantes por columna y era, distribución de clases por temporada).

**EDA de confirmación**
- Distribución H/D/A por temporada, probabilidad de victoria según diferencia de Elo separada por localía, y matriz de correlación.

**Entregables**
- Completar el data card, corregir el PDF de la fase 1 y dejar en este README el comando final que regenera `features.parquet`.

### Detalles técnicos pendientes
- `make_manifest.py`: la URL dice `hhttps://` (typo), y usa una ruta relativa, así que solo funciona corriendo desde la raíz.
- `.gitignore` ignora solo `matches_clean.parquet`. Conviene ignorar `data/interim/`, `data/processed/` y `.pytest_cache/` completas antes de generar `features.parquet`.
- `docs/data_card.md` tiene la sección de cuotas duplicada.
- `src/names_conventions_detections.py` y `src/takes_coutes.py` son scripts de exploración. El primero lee `matches_concat.parquet`, que ya no existe.
