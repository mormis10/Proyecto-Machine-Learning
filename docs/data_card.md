## Uso de columnas como features

Regla: solo se usa información disponible antes del inicio del partido.

**Target:** `FTR` (H / D / A).

**Prohibidas como feature directa** (se conocen al terminar el partido):
- Resultado: `FTHG`, `FTAG`, `FTR`
- Medio tiempo: `HTHG`, `HTAG`, `HTR`
- Estadísticas: `HS`, `AS`, `HST`, `AST`, `HF`, `AF`, `HC`, `AC`, `HY`, `AY`, `HR`, `AR`

Estas columnas solo pueden usarse como agregados de partidos anteriores
(siempre `shift(1)` antes de cualquier `rolling`).

**Excluida:** `Referee` — sin fuga, pero alta cardinalidad y poca señal esperada.

**Cuotas de apuestas:** el mirror de DataHub no las incluye. No se usan como feature;
de conseguirse, solo como benchmark externo.

**Cuotas de apuestas:** no se usan.
- Verificado: ninguno de los 33 CSV de `data/raw/` (93/94–25/26) trae columnas
  de cuotas (p. ej. `B365H`, `PSH`, `AvgH`); solo las 22 columnas de partido.
- Aunque estuvieran disponibles, se excluirían como feature: resumen la opinión
  del mercado y el modelo terminaría imitando a las casas de apuestas en vez de
  aprender del rendimiento de los equipos.
- Uso posible a futuro: benchmark externo ("¿el modelo le gana al mercado?"),
  descargándolas de football-data.co.uk.
