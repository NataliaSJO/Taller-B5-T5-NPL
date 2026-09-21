## Agente sobre informes 10-K: robustez y evaluación

Estado: evaluación incompleta. Informe provisional; no permite concluir qué agente es mejor.

Modelo: openrouter:google/gemini-3.8-flash. Temperatura: 0; máximo de tokens de salida por llamada: 1024; k: 5.

## Diseño y funcionamiento

Se comparan el baseline propio y el agente final con las mismas 20 preguntas: 7 numéricas, 7 extractivas y 6 comparativas. El corpus contiene 48 secciones de seis compañías en dos ejercicios; las comparativas aportan evidencia de ambos años.

El agente consulta get_xbrl_fact para cifras, search_filings para texto, read_section para una sección completa y list_available para comprobar cobertura. El sistema final incorpora filtros, fusión BM25/densa por posiciones y reescritura al inglés. Conserva decimales XBRL, comprueba cifras y permite una corrección antes de abstenerse; limita a ocho llamadas a herramientas y diez al modelo, con reescrituras adicionales registradas.

El acierto exige superar los criterios aplicables de cifra, cita y trayectoria. La cita se comprueba literalmente, con sus metadatos y procedencia, antes del juez semántico. Las preguntas numéricas sin anclas no necesitan juez. Los fallos técnicos se conservan y no se presentan como respuestas evaluadas.

## Comparación de los agentes

| Métrica | Baseline | Final |
| --- | --- | --- |
| Preguntas evaluadas / 20 | 4.0000 | 0.0000 |
| Acierto global | 0.2000 | 0.0000 |
| Acierto numéricas | 0.5714 | 0.0000 |
| Acierto extractivas | 0.0000 | 0.0000 |
| Acierto comparativas | 0.0000 | 0.0000 |
| Recall de la trayectoria | No disponible | No disponible |
| Coste medio agente (USD) | 0.0038 | No disponible |
| Preguntas con coste conocido | 4.0000 | 0.0000 |
| Latencia media (s) | 11.5746 | 4.5005 |
| Llamadas a herramientas | 1.4000 | 1.0000 |

* Mejor valor observado (ambos si empatan), sólo con evaluación completa. Las tasas usan todas las preguntas, incluidos errores y pendientes. Las medias de coste y latencia excluyen valores desconocidos; con ejecución incompleta no describen una comparación válida. El coste del agente incluye reescritura; el juez se contabiliza por separado.

No se interpreta la diferencia de acierto entre agentes porque faltan evaluaciones. Los ceros de preguntas no intentadas no demuestran mala calidad del modelo.

## Medición aislada de la búsqueda

| Variante | Recall@5 | Evidencias |
| --- | --- | --- |
| denso | 0.00% | 19 |
| filtros | 26.32% | 19 |
| hibrido | 26.32% | 19 |
| hibrido_reescrito | 73.68% | 19 |

Los filtros cambian el recall en +26.3 puntos; la fusión híbrida añade +0.0; la reescritura añade +47.4. En esta medición, la fusión no mejoró el resultado del filtrado. Los filtros proceden del golden: esta prueba aísla el buscador y no mide si el agente sabe elegirlos. El recall de la trayectoria agrega todas las búsquedas del agente y no equivale al recall@5.

## Revisión de respuestas y costes

baseline: no_intentado: 15; evaluado: 4; error: 1.

Coste registrado del juez: 0.000000 USD en 0 preguntas con coste de juez informado.

Ejemplo propio-001: ¿Cuál fue el total de activos de Apple al cierre de FY2024?

Respuesta: El total de activos de Apple al cierre del ejercicio fiscal 2024 (a fecha 28 de septiembre de 2024) fue de 364.980.000.000 USD (364.980 millones de dólares).

Criterios: cifra=True; cita=None; trayectoria=True; acierto=True

Incidencia registrada: PaymentRequiredResponseError: This request would exceed your available credits given your current in-flight requests. Retry after in-flight requests settle, or add credits.

final: no_intentado: 19; error: 1.

Coste registrado del juez: 0.000000 USD en 0 preguntas con coste de juez informado.

Incidencia registrada: PaymentRequiredResponseError: This request would exceed your available credits given your current in-flight requests. Retry after in-flight requests settle, or add credits.

## Reproducibilidad y límites

Ejecutar scripts/ejecutar_evaluacion_s2.py con --salida y una carpeta nueva. La clave se introduce de forma privada o mediante OPENROUTER_API_KEY. El notebook permite también ejecutar los bloques en orden. scripts/generar_informe_s2.py regenera este informe a partir de los CSV y las trazas. Las carpetas incluyen configuración, preguntas y hashes.

La validación del golden es local contra el corpus, no el validador externo del aula. Las 10 preguntas ciegas no están disponibles y su resultado y diferencia frente al golden quedan pendientes. Una muestra de 20 preguntas y un juez automático no garantizan generalización; se conservan respuestas y trazas para revisión humana.

Fuentes locales: resultados\s2\verificacion_20260921\comparacion ; resultados\s2\retrieval\20260921T114707Z_78cf7608
