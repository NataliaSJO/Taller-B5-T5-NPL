# Golden set propio

`golden_set_propio.jsonl` contiene 20 preguntas con respuesta conocida:

| Familia | Cantidad | Identificadores |
| --- | --- | --- |
| Numérica | 7 | `propio-001` a `propio-007` |
| Extractiva | 7 | `propio-008` a `propio-014` |
| Comparativa | 6 | `propio-015` a `propio-020` |

Se incluyen NVIDIA, Microsoft, Apple, Alphabet, Meta y Amazon, y los
ejercicios FY2024 y FY2025. Las preguntas son distintas de las tres del
fichero de ejemplo. El campo `autor` contiene `grupo-propio`: sustituidlo
por el identificador que os hayan asignado en clase.

## Fuentes y formato

Se conservan todos los campos del ejemplo. Las cifras y unidades proceden
de `corpus/xbrl_facts.parquet`; las citas, de `corpus/secciones.jsonl`.
Cada ancla es una frase literal de un máximo de 40 palabras. Los campos
`ancla_inicio` y `ancla_fin` son posiciones de caracteres dentro del texto
de la sección, con inicio incluido y final excluido, como en Python.

`chunk_id_esperado` identifica un fragmento actual que contiene la frase.
La referencia principal es `ancla_texto`: seguirá siendo válida si se
cambia el troceado del corpus.

## Comparativas

Cada comparativa pide un importe de FY2025, su comparación con FY2024 y
una comparación cualitativa apoyada en ambos informes. Se necesitan
`get_xbrl_fact` y `search_filings` para responder todas sus partes.

El campo principal `cifra_esperada` contiene el importe de FY2025, que es
lo que se pregunta primero; no contiene el porcentaje de crecimiento.
Se añaden dos campos sin cambiar los del ejemplo:

- `comparacion`: valores, conceptos XBRL y ejercicios de ambos lados,
  diferencia absoluta y variación porcentual.
- `evidencias_por_ejercicio`: una ancla con su ubicación para FY2024 y otra
  para FY2025. El ancla principal del esquema es la de FY2025.

La variación porcentual es `(valor_2025 - valor_2024) / valor_2024 * 100`.
Se guarda con seis decimales y se resume con dos en la respuesta en prosa.
El validador original admite estos campos adicionales. Los evaluadores
iniciales del baseline todavía no comprueban todos sus componentes;
acertar solo el importe de FY2025 no demuestra una comparación correcta.

La pregunta de beneficio diluido por acción conserva los decimales de XBRL
(7,46 USD/shares), para que permita detectar redondeos indebidos del agente
o de la herramienta.

## Comprobaciones realizadas

El conjunto pasa `validar(preguntas, exigir_20=True)` del material de clase.
Además, se han contrastado las 13 cifras principales con XBRL, los dos
valores y los cálculos de las seis comparativas, y las 19 anclas con sus
secciones, posiciones y fragmentos. No se ha llamado al modelo para crear
o validar este archivo.

En el notebook, una vez ejecutada la celda que define `validar`, se puede
comprobar de nuevo sin consumir API:

```python
ruta = RAIZ / "src/golden_set_propio.jsonl"
preguntas = [json.loads(linea) for linea in ruta.read_text(
    encoding="utf-8").splitlines() if linea.strip()]
problemas = validar(preguntas, exigir_20=True)
assert not problemas, "\n".join(problemas)
print("20 preguntas validadas.")
```
