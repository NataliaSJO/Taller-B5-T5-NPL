# Comprobación local de S2 — 21 de septiembre de 2026

Se ejecutó completo `src/S2_Robustez_y_Evaluacion_Alumno.ipynb` con el kernel
`taller-10k`, sin llamadas a OpenRouter. Se guardaron las salidas del notebook.
Las 13 pruebas de `python -m unittest discover -s tests -v` pasaron, incluidos
los límites de llamadas, la corrección y abstención dentro del grafo real,
la carga del baseline propio y los evaluadores. Los modelos de las pruebas
son locales y no cuentan como resultados de evaluación del LLM.

El conjunto propio pasó la validación local: 20 preguntas, 6 comparativas y
19 anclas entre ambos ejercicios. No se ha ejecutado un validador oficial
externo, que no está incluido en los materiales del repositorio.

Resultados reales del buscador con metadatos conocidos:

| Variante | Aciertos / anclas | Recall@5 |
| --- | ---: | ---: |
| Denso sin filtros | 0 / 19 | 0,00 % |
| Denso con filtros | 5 / 19 | 26,32 % |
| BM25 + denso con filtros | 5 / 19 | 26,32 % |

Detalle y configuración en
`retrieval/20260921T110329Z_27041fea/`. El híbrido no mejora el agregado respecto
al filtro en esta medición. Este resultado no mide la capacidad del agente
de inferir los filtros: se toman del golden.

Pendiente de ejecución con API: reescritura de consultas, respuestas de ambos
agentes sobre las 20 preguntas y juicio semántico de citas. El notebook contiene
el código para generarlos y guardarlos al activar `EJECUTAR_API = True`.
No hay resultados finales del LLM ni costes inventados.

## Recuperación posterior del notebook

Se recuperó la versión de `src` después de detectar que contenía otra vez los
ejercicios de clase sin completar. Los nueve bloques de código vuelven a estar
visibles, con explicaciones `CAMBIO S2`. Se conservaron los apuntes añadidos como
referencia y se guardaron las copias anterior, recuperada y verificada en
`recuperacion/20260921T112815Z_e846188d/`.

Se ejecutó nuevamente el notebook completo sin API y sin errores. Las **14
pruebas locales** pasaron, incluida la comprobación nueva que detecta si el
notebook pierde sus bloques adaptados o deja de coincidir con el módulo.
La nueva medición del retrieval reproduce la tabla anterior y está en
`retrieval/20260921T113115Z_c57ae22c/`. La evaluación con API sigue pendiente.

La recuperación se puede repetir con
`python scripts/recuperar_notebook_s2.py`; guarda copia previa y se detiene si
el notebook ya está adaptado para no borrar modificaciones posteriores.
