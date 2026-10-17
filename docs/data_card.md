# Data card: Premier League Match Predictor

Documenta la fuente de los datos, las decisiones de preparación y su justificación.

## Fuente y versiones

- **Fuente:** 33 CSV de DataHub, una por temporada (1993/94 a 2025/26), en `data/raw/`. Esos archivos no se editan nunca.
- **Trazabilidad:** `data/raw/MANIFEST.csv` registra, por archivo, URL, fecha de descarga, SHA-256 y número de filas.
  Si un CSV cambia, su hash lo delata. Se regenera con `src/make_manifest.py`.
- **Riesgo:** la fuente es un solo espejo. El manifiesto permite detectar cambios, pero no recuperar datos si el espejo desaparece.

## Limpieza

Resultado: `data/interim/matches_clean.parquet`, con 12,704 partidos (462 en 93/94 y 94/95, 380 en el resto) y 24 columnas
(las 22 de partido de los CSV más `Season` y `SeasonStart`).

- `Date` se parsea aceptando ISO, `dd/mm/yyyy` y `dd/mm/yy`; si aparece otro formato, falla en lugar de adivinar.
- Los nombres de equipo se normalizan con `src/team_names.py` (51 equipos, nombres canónicos = los de DataHub).
- Se ordena por `["Date", "HomeTeam"]` con índice 0..12703. Ese índice es el `match_id` con el que todas las features vuelven a unirse.

## Uso de columnas como features

Regla: solo se usa información disponible antes del inicio del partido.

**Target:** `FTR` (H / D / A).

**Prohibidas como feature directa** (se conocen al terminar el partido):
- Resultado: `FTHG`, `FTAG`, `FTR`
- Medio tiempo: `HTHG`, `HTAG`, `HTR`
- Estadísticas: `HS`, `AS`, `HST`, `AST`, `HF`, `AF`, `HC`, `AC`, `HY`, `AY`, `HR`, `AR`

Estas columnas solo pueden usarse como agregados de partidos anteriores
(siempre `shift(1)` antes de cualquier `rolling`).

**Excluida:** `Referee`. No hay fuga, pero tiene alta cardinalidad y se espera poca señal.

**Cuotas de apuestas:** no se usan.
- Verificado: ninguno de los 33 CSV de `data/raw/` (93/94-25/26) trae columnas de cuotas (p. ej. `B365H`, `PSH`, `AvgH`);
  solo las 22 columnas de partido.
- Aunque estuvieran disponibles, se excluirían como feature: resumen la opinión del mercado y el modelo
  terminaría imitando a las casas de apuestas en vez de aprender del rendimiento de los equipos.
- Uso posible a futuro: benchmark externo ("¿el modelo le gana al mercado?"), descargándolas de football-data.co.uk.

## Burn-in

Las temporadas 1993/94-1999/00 se usan solo para inicializar el Elo y la forma reciente; no entran a entrenamiento
ni evaluación. En 1993/94 la dispersión del Elo es 44 puntos y el favorito gana 49%; desde ~1999/00 se estabiliza (≈73, 60%).
Además, las estadísticas de tiros (`HS`, `AS`, `HST`, `AST`) faltan al 100% antes de 2000/01 y al 0% desde entonces.

Regla: las features se calculan sobre todas las temporadas y se filtra al final (`TRAIN_START = 2000`).
Filtrar antes rompe el Elo sin dar error.

## Features

Todas se calculan en `src/features.py` y usan solo partidos anteriores.

| Grupo | Columnas | Detalle |
|---|---|---|
| Elo | `elo_home`, `elo_away`, `elo_diff` | Valor **antes** del partido. 1500 inicial, K = 20, +60 de localía, 80% de arrastre entre temporadas, ascendidos con el promedio de los descendidos. |
| Forma | `form_points/gf/ga`, `form_venue_points/gf/ga` | Promedio de los últimos 5 partidos; la versión `venue` solo cuenta partidos en la misma condición (local o visitante). La ventana cruza temporadas. |
| Tiros | `roll_shots_for/against`, `roll_sot_for/against` | Promedio de los últimos 5 partidos (tiros y tiros a puerta). |
| Diferencias | `diff_*` | `home_x - away_x` para las 10 variables anteriores. |
| Bandera | `home_pocos_partidos`, `away_pocos_partidos` | 1 si el equipo aún no tiene 5 partidos previos de resultados o de tiros. |

Columnas que reciben ambos modelos (`MODEL_COLS`): `elo_diff`, las 10 `diff_*` y las dos banderas.
No se incluyen `home_x` ni `away_x` junto con `diff_x`: `diff_x` es exactamente `home_x - away_x`, y meter las tres
sería colinealidad perfecta para Logistic Regression.

## Valores faltantes e imputación

Logistic Regression no acepta `NaN`, y quedan en tres situaciones: el primer partido de cada equipo,
el primer partido en cada sede, y los equipos sin partidos previos con estadísticas de tiros
(por ejemplo Brentford en 2021 o Luton en 2023), además de la primera jornada de 2000/01.

Decisión:
1. **Bandera `pocos_partidos`.** Marca los casos con historia incompleta para que un valor imputado no pase por real.
2. **Valor previo.** Los `NaN` se rellenan con la media de la liga calculada con fechas **estrictamente anteriores** al partido
   (por sede en `form_venue_*`). Se calcula después del `rolling`, nunca antes. No se usan partidos del mismo día porque son simultáneos.
3. **Descarte.** Los partidos que siguen sin valor previo se descartan: son los 9 de la primera jornada de 2000/01,
   donde no existe ningún partido anterior con tiros. El código solo tolera esto en los primeros 14 días de datos;
   en cualquier otro lugar falla con un error.


| Variable | 93/94-99/00 | 00/01-09/10 | 10/11-19/20 | 20/21-25/26 |
|---|---|---|---|---|
| `diff_form_points` | 0.78 | 0.26 | 0.16 | 0.09 |
| `diff_form_venue_points` | 1.56 | 0.53 | 0.32 | 0.18 |
| `diff_roll_shots_for` | 100.00 | 0.66 | 0.18 | 0.13 |

Después de imputar, las variables finales tienen 0% de faltantes en todas las eras.

Limitación: un equipo recién ascendido suele ser más débil que la media de la liga, así que su valor previo es optimista.
La bandera y el Elo (los ascendidos arrancan con el promedio de los descendidos) lo compensan en parte.

## Resultado final

`data/processed/features.parquet`: **9,871 partidos** (2000/01-2025/26) y **39 columnas**
(4 de contexto, 3 de Elo, 22 `home_*`/`away_*` y 10 diferencias).

## Particiones

Validación walk-forward (`src/splits.py`): se prueba en cada temporada de 2020/21 a 2025/26 y se entrena con todo lo anterior.
Dentro del bloque de entrenamiento, la última temporada es la validación interna para hiperparámetros; nunca se usa la de prueba.
Cada fold verifica con un `assert` que `max(fecha_train) < min(fecha_test)`.

## Temporada atípica: 2020/21

Se jugó sin público. Es la única temporada en que ganan más los visitantes (40.3%) que los locales (37.9%).
Se marca como atípica en los folds (`test_atypical`, `val_atypical`) pero no se excluye; excluirla o no queda como decisión de la Fase 3.
Al probar en 2021/22, la validación interna cae en 2020/21.

## Distribución de clases

| Período | Partidos | H (%) | D (%) | A (%) |
|---|---|---|---|---|
| 2000/01-2009/10 | 3,791 | 47.1 | 25.7 | 27.1 |
| 2010/11-2019/20 | 3,800 | 45.7 | 24.6 | 29.7 |
| 2020/21-2025/26 | 2,280 | 43.1 | 23.6 | 33.3 |
| **Total** | **9,871** | **45.6** | **24.8** | **29.5** |

Las clases están desbalanceadas: predecir siempre "gana el local" acierta el 45.6%, que es el piso de referencia
para cualquier modelo. El empate es la clase minoritaria.

## Hallazgos del análisis exploratorio

Figuras en `docs/figures/` (generadas por `src/eda.py`).

- **Ventaja de local.** Medida como %H - %A, fue negativa solo en 2020/21 (-2.4 puntos). Entre 2000/01 y 2019/20 nunca bajó de
  10.8 puntos (mínimo en 2015/16); en 2024/25 llegó a 6.1. La variación entre temporadas es grande, por lo que no se interpreta como tendencia firme.
- **Elo y probabilidad de ganar.** La proporción de victorias sube de forma continua con la diferencia de Elo (propio menos rival),
  desde cerca de 0 hasta más de 0.85. Con igual Elo, el local gana más que el visitante en todo el rango
  (entre 41% y 49% contra entre 25% y 29% en los intervalos centrales). La brecha entre ambos es de 15 a 20 puntos porcentuales
  en la zona central y de 10 o menos en los extremos, donde las probabilidades se acercan a 0 y a 1. Eso no basta para afirmar una interacción
  entre localía y Elo.
- **Correlaciones.** Solo cuatro pares superan 0.7: tiros y tiros a puerta a favor (0.83), tiros y tiros a puerta en contra (0.81),
  puntos y goles a favor por sede (0.74), y puntos y goles a favor (0.73). Ninguno se acerca a 1, por lo que no se eliminan columnas.
  Las banderas casi no se correlacionan con el resto.

## Calidad

- Tests de recálculo y de perturbación para Elo, forma, tiros, bandera y valor previo (`tests/`).
- `src/quality_report.py` genera `docs/quality_report.txt`: faltantes por columna y era, partidos descartados, clases por temporada.
- `src/eda.py` genera `docs/figures/`: H/D/A por temporada, victoria según diferencia de Elo por localía y matriz de correlación.