# Comprobación local de S2 — 22 de septiembre de 2026

Se ejecuta sin llamadas a OpenRouter con:

```powershell
python -m unittest discover -s tests -v
```

Pasan las **28 pruebas**, que cubren el contrato de las cuatro firmas, los
límites de llamadas, la corrección y la abstención dentro del grafo real, el
corte por tiempo, la carga del baseline propio, los tres evaluadores, la
reparación literal de citas sobre un caso real del corpus, el filtrado de
fechas y viñetas del extractor de cifras y la estimación de coste por tarifa.
Los modelos de las pruebas son locales y no cuentan como resultados de LLM.

El conjunto propio pasa la validación local: 20 preguntas, 6 comparativas y
19 anclas entre ambos ejercicios. No se ha ejecutado un validador oficial
externo, que no está incluido en los materiales del repositorio.

## Clon limpio

El enunciado exige que `responder()` y `evaluar()` funcionen sobre un clon
limpio sin editar nada. Comprobado copiando el árbol de trabajo sin
`resultados/` ni artefactos derivados: **15,7 s** hasta tener el sistema en pie,
incluyendo la extracción de los dos ZIP, la verificación de hashes contra los
manifiestos y la regeneración automática de `corpus/derivado/etiquetas.parquet`.
Contrato verificado en ese clon: `search_filings(query, ticker, fiscal_year,
item, k)`, `get_xbrl_fact(ticker, fiscal_year, concept)`, `read_section(ticker,
fiscal_year, item)`, `list_available()`, y los ocho campos obligatorios del
esquema presentes (sólo se añaden `datos` y `citas`).

## Buscador sin API

Medición con los filtros tomados del golden; es una cota superior, porque el
agente tiene que inferirlos. Con la consulta en español el índice denso que se
entrega no recupera nada:

| Variante | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| denso | 0,0 % | 0,0 % | 0,0 % | 0,0 % |
| filtros | 5,3 % | 10,5 % | 26,3 % | 47,4 % |
| hibrido | 5,3 % | 15,8 % | 26,3 % | 36,8 % |
| hibrido_enc | 0,0 % | 15,8 % | 21,1 % | 36,8 % |

La fusión BM25 no mejora el recall@5 sobre el filtrado y lo empeora a k=10; la
etiqueta de encabezado también empeora mientras la consulta va en español. La
otra mitad de la matriz, con la consulta reescrita, necesita API y está en
`RESUMEN.md`.

## Recuperación del notebook

Si el notebook vuelve a sobrescribirse con la copia de clase:

```powershell
python scripts/recuperar_notebook_s2.py
```

Guarda copia previa y se detiene si el notebook ya está adaptado, para no
borrar modificaciones posteriores. La prueba
`test_notebook_adaptado_y_sincronizado` detecta la sustitución.
