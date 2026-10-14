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
