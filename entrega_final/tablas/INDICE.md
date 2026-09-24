# Tablas generadas por el código

Copias sin editar de las salidas del código. Cada fila indica la carpeta de origen en `resultados/` y el comando que la genera.

| Archivo | Contenido | Origen | Comando |
| --- | --- | --- | --- |
| `00_RESUMEN.md` | Resumen baseline frente a final, cambios respecto al 21-sep y costes | `resultados/RESUMEN.md` | `scripts/generar_resumen.py --anterior` |
| `01_baseline_vs_final.csv` / `.md` | Tabla principal: acierto, recall, coste, latencia y llamadas | `resultados/s2/verificacion_20260924_opus/comparacion/` | `scripts/ejecutar_evaluacion_s2.py` |
| `01_mcnemar.json` | Prueba de McNemar exacta (12 a 0, p = 0,0005) | ídem | ídem |
| `02_*_por_familia.csv` | Acierto por familia, baseline y final | ídem, `baseline/` y `final/` | ídem |
| `02_*_por_pregunta.csv` | Métricas por pregunta de cada evaluador | ídem | ídem |
| `03_recall_at_5.csv` | Recall@5 de las variantes de búsqueda | `resultados/s2/retrieval/20260924T084746Z_7fb3e330/` | `medir_retrieval()` |
| `03_recall_por_k.csv` | Recall a k = 1, 3, 5 y 10 | ídem | ídem |
| `04_panel_modelos.csv` | Otros modelos como agente, juez Claude Opus 5.5 (Gemini 3.8 Flash es la fila `final` de `01`) | `resultados/s2/panel_modelos/20260924_juez_opus/` | `scripts/comparar_panel_modelos.py` |
| `05_golden_oficial_*.csv` | Golden oficial de clase: 10/20 | `resultados/s2/golden_oficial/20260924T101726Z_d2a8d585/` | `evaluar("Clase_2/golden_set.jsonl", salida=...)` |
| `06_ciegas_*.csv` | Preguntas ciegas: 4/10 | `resultados/s2/holdout/20260924_holdout/` | `evaluar("holdout.jsonl", salida=...)` |
| `06_ciegas_reintento_ho003.csv` | Repetición de ho-003 tras el error 429: acierta | `resultados/s2/holdout/20260924_reintento_ho003/` | ídem, sólo ho-003 |
| `07_escala_sp500.json` | Coste medido y estimación para el S&P 500 | `resultados/s2/escala/20260924T094937Z_a8ac7579/` | `scripts/estimar_escala.py` |

Las respuestas completas con sus trazas (`respuestas.jsonl`) y la configuración de cada ejecución están en las carpetas de origen.
