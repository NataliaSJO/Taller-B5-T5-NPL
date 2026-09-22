## Agente sobre informes 10-K: robustez y evaluación

Estado: evaluación incompleta. Informe provisional; no permite concluir qué agente es mejor.

Modelo: openrouter:google/gemini-3.8-flash. Temperatura: 0; máximo de tokens de salida por llamada: 4096; k: 5.

## Diseño y funcionamiento

Se comparan el baseline propio y el agente final con las mismas 20 preguntas: 7 numéricas, 7 extractivas y 6 comparativas. El corpus contiene 48 secciones de seis compañías en dos ejercicios; las comparativas aportan evidencia de ambos años.

El agente consulta get_xbrl_fact para cifras, search_filings para texto, read_section para una sección completa y list_available para comprobar cobertura. El sistema final incorpora filtros, fusión BM25/densa por posiciones, etiqueta de encabezado y reescritura al inglés. Conserva decimales XBRL, repara la cita al tramo literal del fragmento, comprueba cifras y permite una corrección antes de abstenerse; limita a 8 llamadas a herramientas y 12 al modelo por pregunta, con un corte de tiempo y las reescrituras registradas.

El acierto exige superar los criterios aplicables de cifra, cita y trayectoria. El criterio de cita es el del enunciado —que exista y respalde lo que se afirma—; que además caiga sobre la frase anclada en el golden se reporta aparte, porque eso mide recuperación y penalizaba pasajes distintos igualmente válidos. La cita se comprueba literalmente, con sus metadatos y procedencia, antes del juez semántico. Las preguntas numéricas sin anclas no necesitan juez. Los fallos técnicos se conservan y no se presentan como respuestas evaluadas.

## Comparación de los agentes

| Métrica | Baseline | Final |
| --- | --- | --- |
| Preguntas evaluadas / 20 | 20.0000 | 19.0000 |
| Acierto global | 0.5500 | 0.9500 |
| Acierto numéricas | 0.8571 | 1.0000 |
| Acierto extractivas | 0.7143 | 0.8571 |
| Acierto comparativas | 0.0000 | 1.0000 |
| Recall de la trayectoria | 0.7692 | 0.8750 |
| Coste medio agente (USD) | 0.0145 | 0.0155 |
| Preguntas con coste conocido | 20.0000 | 20.0000 |
| Latencia media (s) | 24.2622 | 29.3904 |
| Llamadas a herramientas | 3.7500 | 3.1500 |

* Mejor valor observado (ambos si empatan), sólo con evaluación completa. Las tasas usan todas las preguntas, incluidos errores y pendientes. Las medias de coste y latencia excluyen valores desconocidos; con ejecución incompleta no describen una comparación válida. El coste del agente incluye reescritura; el juez se contabiliza por separado.

No se interpreta la diferencia de acierto entre agentes porque faltan evaluaciones. Los ceros de preguntas no intentadas no demuestran mala calidad del modelo.

## Medición aislada de la búsqueda

Esta medición se conserva como experimento separado: modelo openrouter:google/gemini-3.8-flash, máximo de salida 4096 tokens. La comparación de agentes utiliza 4096 tokens. No se han repetido las consultas de este experimento al cambiar el límite de salida de los agentes.

| Variante | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Evidencias |
| --- | --- | --- | --- | --- | --- |
| denso | 0.00% | 0.00% | 0.00% | 0.00% | 19 |
| filtros | 5.26% | 10.53% | 26.32% | 47.37% | 19 |
| hibrido | 5.26% | 15.79% | 26.32% | 36.84% | 19 |
| hibrido_enc | 0.00% | 15.79% | 21.05% | 36.84% | 19 |
| denso_reescrito | 15.79% | 31.58% | 36.84% | 47.37% | 19 |
| filtros_reescrito | 36.84% | 63.16% | 63.16% | 84.21% | 19 |
| hibrido_reescrito | 36.84% | 57.89% | 63.16% | 100.00% | 19 |
| hibrido_enc_reescrito | 31.58% | 68.42% | 73.68% | 84.21% | 19 |

La matriz separa las dos mejoras en lugar de encadenarlas, que es lo que permite atribuirlas. Sobre la consulta original, los filtros cambian el recall@5 en +26.3 puntos y la fusión BM25 en +0.0. Reescribir la consulta cambia el filtrado en +36.8 puntos y el híbrido en +36.8: la reescritura es, sobre todo, traducción, porque las preguntas van en español y el corpus está en inglés. La etiqueta de encabezado añade +10.5 puntos sobre el híbrido reescrito. Los filtros proceden del golden: esta prueba aísla el buscador y no mide si el agente sabe elegirlos. El recall de la trayectoria agrega todas las búsquedas del agente y no equivale al recall@5.

## Revisión de respuestas y costes

baseline: evaluado: 20.

Coste registrado del juez: 0.001361 USD en 5 preguntas con coste de juez informado.

Ejemplo propio-007: ¿Cuál fue el beneficio diluido por acción de Apple en FY2025, expresado en dólares por acción?

Respuesta: El beneficio diluido por acción (EarningsPerShareDiluted) de Apple en el ejercicio fiscal 2025 (FY2025) fue de 7,00 USD por acción.

Criterios: cifra=False; cita=None; trayectoria=True; acierto=False

Ejemplo propio-010: ¿Por qué una interrupción en el suministro de componentes críticos puede ser especialmente perjudicial para Apple, según su informe de FY2025?

Respuesta (extracto): Según el informe 10-K de Apple para el ejercicio fiscal 2025 (Item 1A), una interrupción en el suministro de componentes críticos puede ser especialmente perjudicial por las siguientes razones:

1. **Dependencia de fuentes únicas o limitadas (*single-source*):** Apple obtiene ciertos componentes de fuentes únicas o limitadas, y sus nuevos productos frecuentemente requieren componentes personalizados que están disponibles a través de un solo proveedor.
2. **Dificultad de reemplazo y costes de transición:** Identificar y cualificar fuentes alternativas o realizar cambios en la cadena de suministro requiere un tiempo y recursos considerables, además de acarrear importantes riesgos regulatorios …

Criterios: cifra=None; cita=False; trayectoria=True; acierto=False

La cita no superó las comprobaciones previas de literalidad, metadatos o evidencia recuperada; no se solicitó juicio semántico.

final: evaluado: 19; error: 1.

Coste registrado del juez: 0.003561 USD en 12 preguntas con coste de juez informado.

Ejemplo propio-001: ¿Cuál fue el total de activos de Apple al cierre de FY2024?

Respuesta: El total de activos de Apple al cierre de FY2024 fue de 364980000000 USD.

Criterios: cifra=True; cita=None; trayectoria=True; acierto=True

Incidencia registrada: RuntimeError: El agente terminó sin respuesta estructurada; puede haber alcanzado un límite

## Reproducibilidad y límites

Ejecutar scripts/ejecutar_evaluacion_s2.py con --salida y una carpeta nueva. La clave se introduce de forma privada o mediante OPENROUTER_API_KEY. El notebook permite también ejecutar los bloques en orden. scripts/generar_informe_s2.py regenera este informe a partir de los CSV y las trazas. Las carpetas incluyen configuración, preguntas y hashes.

La validación del golden es local contra el corpus, no el validador externo del aula. Las 10 preguntas ciegas no están disponibles y su resultado y diferencia frente al golden quedan pendientes. Una muestra de 20 preguntas y un juez automático no garantizan generalización; se conservan respuestas y trazas para revisión humana.

La latencia depende de la red, la carga del proveedor y la caché local. Las versiones se ejecutan secuencialmente, primero baseline y después final; no es un ensayo aleatorizado de rendimiento. Los extractos del informe no sustituyen las respuestas completas guardadas en respuestas.jsonl.

Fuentes locales: resultados\s2\comparacion\20260921T222953Z_e54cd3c9 ; resultados\s2\retrieval\20260921T215557Z_b69136ba
