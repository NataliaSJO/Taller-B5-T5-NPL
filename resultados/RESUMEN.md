# Resultados de la evaluación

Modelo del agente: `openrouter:google/gemini-3.8-flash`, máximo 4096 tokens de salida, temperatura 0. Tolerancia numérica 1%. Límite de 8 llamadas a herramienta y 12 al modelo por pregunta.

Modelos en las llamadas registradas (`respuestas.jsonl` del final): agente y reescritura, `google/gemini-3.8-flash`; juez de citas, `anthropic/claude-opus-5.5`. `configuracion.json` de esta carpeta sólo guarda el modelo del agente.

Carpeta de resultados: `resultados/s2/verificacion_20260924_opus/comparacion`.

## Baseline frente a sistema final

| Métrica | Baseline | Final |
| --- | --- | --- |
| Preguntas evaluadas | 20 | 20 |
| Acierto global | 30.0% | **90.0%** |
| Acierto numéricas | 85.7% | **100.0%** |
| Acierto extractivas | 0.0% | **85.7%** |
| Acierto comparativas | 0.0% | **83.3%** |
| Cita válida | 0.0% | **84.6%** |
| Cita sobre el ancla del golden | 46.2% | **92.3%** |
| Recall@5 del buscador (aislado) | 26.3% | **68.4%** |
| Recall de la trayectoria | 92.3% | **100.0%** |
| Coste medio del agente (USD) | 0.0166 | **0.0157** |
| Coste medio estimado del agente (USD) | 0.0166 | **0.0157** |
| Coste del juez por pregunta (USD) | 0.0034 | 0.0053 |
| Coste total del juez (USD) | 0.0674 | 0.1063 |
| Latencia media (s) | **27.0** | 27.1 |
| Llamadas a herramienta | 3.85 | **3.05** |
| Avisos del guardrail | 0 | 0 |

El coste del agente incluye la reescritura de consultas. El del juez es sólo de la evaluación: no forma parte del agente en uso.

Prueba pareada (McNemar exacto) sobre las mismas 20 preguntas: sólo acierta el final en 12, sólo el baseline en 0, p = 0.0005. Con 20 preguntas el intervalo de confianza de una tasa ronda ±20 puntos: la tabla se lee junto a esta prueba, no en su lugar.

## Buscador: matriz de mejoras

| Variante | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Evidencias |
| --- | --- | --- | --- | --- | --- |
| denso | 0.0% | 0.0% | 0.0% | 0.0% | 19 |
| filtros | 5.3% | 10.5% | 26.3% | 47.4% | 19 |
| hibrido | 5.3% | 15.8% | 26.3% | 36.8% | 19 |
| hibrido_enc | 0.0% | 15.8% | 21.1% | 36.8% | 19 |
| denso_reescrito | 15.8% | 21.1% | 26.3% | 47.4% | 19 |
| filtros_reescrito | 26.3% | 57.9% | 63.2% | 84.2% | 19 |
| hibrido_reescrito | 31.6% | 57.9% | 73.7% | 94.7% | 19 |
| hibrido_enc_reescrito | 31.6% | 68.4% | 68.4% | 73.7% | 19 |

Los filtros salen del golden: la medición aísla el buscador y es una cota superior de lo que consigue el agente, que debe inferirlos de la pregunta.
Carpeta: `resultados/s2/retrieval/20260924T084746Z_7fb3e330`.

## Cambios frente a una ejecución anterior

Ejecución anterior: `resultados/s2/comparacion/20260921T230019Z_c0888bbc`. Mismas 20 preguntas y mismos evaluadores de código. Si cambia el juez, cambia la vara de medir: las diferencias no son sólo del agente.

| Versión | Acierto antes | Acierto ahora | Juez antes | Juez ahora |
| --- | --- | --- | --- | --- |
| baseline | 55.0% | 30.0% | `google/gemini-3.5-flash-lite` | `anthropic/claude-opus-5.5` |
| final | 95.0% | 90.0% | `google/gemini-3.5-flash-lite` | `anthropic/claude-opus-5.5` |

Preguntas que cambian de resultado:

| Versión | Pregunta | Familia | Antes | Ahora | Evaluador que cambia |
| --- | --- | --- | --- | --- | --- |
| baseline | propio-008 | extractiva | acierto | fallo | cita |
| baseline | propio-009 | extractiva | acierto | fallo | cita |
| baseline | propio-012 | extractiva | acierto | fallo | cita |
| baseline | propio-013 | extractiva | acierto | fallo | cita |
| baseline | propio-014 | extractiva | acierto | fallo | cita |
| final | propio-011 | extractiva | acierto | fallo | cita |
| final | propio-015 | comparativa | fallo | acierto | cita |
| final | propio-018 | comparativa | acierto | fallo | cita |

## Varianza entre ejecuciones

El **mismo** agente baseline, las mismas preguntas y los mismos evaluadores, medido dos veces. La temperatura es 0, pero ni la API ni el juez son deterministas.

Ejecución 1: `resultados/s2/comparacion/20260921T220600Z_a1e83bd5`, juez `google/gemini-3.5-flash-lite`. Ejecución 2: `resultados/s2/comparacion/20260921T230019Z_c0888bbc`, juez `google/gemini-3.5-flash-lite`.

| Familia | Ejecución 1 | Ejecución 2 |
| --- | --- | --- |
| Global | 45.0% | 55.0% |
| comparativa | 0.0% | 0.0% |
| extractiva | 28.6% | 71.4% |
| numerica | 100.0% | 85.7% |

Cambian de resultado **4 de 20 preguntas**. Una de ellas se explica por el reintento de firma de pensamiento que incorpora la segunda ejecución; el resto es ruido. Con 7 preguntas por familia, las cifras por familia se mueven decenas de puntos sin que cambie nada: no se deben leer como diferencias reales.

## Control de memorización (sin herramientas)

Las mismas preguntas contra el mismo modelo, sin corpus ni herramientas. Mide cuánto del acierto no depende del 10-K: el enunciado penaliza acertar por el camino equivocado.

| Medida | Valor |
| --- | --- |
| Cifras acertadas de memoria | 1/13 (7.7%) |
| Abstenciones | 13/20 |
| Citas inventadas (no existen en el corpus) | 2/2 |
| Coste del control (USD) | 0.0521 |

## Huecos reales del XBRL

Conceptos que una compañía no reporta (Amazon no publica GrossProfit, Liabilities ni ResearchAndDevelopmentExpense; Meta y Alphabet tampoco GrossProfit). La respuesta correcta es decir que no está. El golden propio no cubre este caso, así que se mide aparte.

| Ejecución | Se abstiene correctamente | Coste (USD) |
| --- | --- | --- |
| 20260921T225238 | 1/3 | 0.0814 |
| 20260921T225752 | 3/3 | 0.0163 |

## Cómo se regenera

```powershell
python -m unittest discover -s tests -v
python scripts/ejecutar_evaluacion_s2.py --salida resultados\s2\<carpeta>
python scripts/generar_resumen.py --comparacion <carpeta>\comparacion --retrieval resultados\s2\retrieval\<marca> [--anterior <otra comparación>] [--repeticion <A> --repeticion-de <B>]
```
