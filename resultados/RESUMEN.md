# Evaluación del baseline con el agente del notebook

Se ha utilizado el objeto `agente` de la sesión abierta de
`src/Baseline_Agente_10K.ipynb`, sin reconstruirlo ni copiar la clave de API.
El modelo de la sesión era `openrouter:google/gemini-3.8-flash`, con un
máximo de 1.024 tokens de salida por llamada.

## Estado

- El golden set pasa el validador de clase con `exigir_20=True`:
  20 preguntas, de las cuales 6 son comparativas.
- Se intentaron las 20 preguntas y se guardó un registro para cada una.
- Las 7 preguntas numéricas devolvieron respuesta estructurada.
- Las 13 preguntas extractivas y comparativas no llegaron a completar
  su respuesta: OpenRouter rechazó peticiones por crédito disponible y,
  posteriormente, por el límite de 20 solicitudes por minuto de la cuenta.
- Después de esperar se reintentó `propio-008`; volvió a producirse un
  `PaymentRequiredResponseError`. Ese intento también está conservado.

La evaluación está **parcial**, no completada satisfactoriamente para las
20 preguntas. Los errores de infraestructura no deben interpretarse como
respuestas incorrectas del agente ni ocultarse al comparar sistemas.

## Resultados guardados

La ejecución principal está en:

`baseline/20260917T180626Z_938f35c6/`

Contiene `respuestas.jsonl`, `metricas.csv`, `resumen_por_familia.csv`,
`configuracion.json`, `preguntas.jsonl` y las copias de `baseline.ipynb`
y `miax_s1.py` utilizadas. El reintento está en:

`baseline/20260917T180958Z_346a12b8/`

El coste comunicado por OpenRouter para las 7 respuestas completas suma
**0,05130825 USD**. Es un subtotal conocido: las llamadas intermedias de
ejecuciones fallidas pueden tener consumo que estas métricas no recogen.

En las 7 preguntas numéricas, 6 pasan la comprobación conjunta de cifra
y unidad. `propio-007` devuelve la cifra correcta, 7,46, pero usa la unidad
`USD/acción` en lugar de `USD/shares`, y el comparador literal la rechaza.
La respuesta también recurre al texto tras encontrar el valor redondeado
de la herramienta. Son observaciones para revisar en la siguiente sesión;
no se han alterado las respuestas ni las reglas para mejorar la puntuación.

Las comprobaciones son las iniciales del baseline. No demuestran todavía
que cada cita respalde toda la respuesta ni que las comparativas se hayan
resuelto completamente. Las medias de respuestas y costes excluyen valores
desconocidos; sus columnas `count` muestran cuántos casos sustentan cada media.

## Reanudar las preguntas pendientes

`validacion/preguntas_pendientes.jsonl` contiene las 13 preguntas sin
respuesta completa. Cuando se haya resuelto la disponibilidad de crédito,
se pueden evaluar con el mismo agente de la sesión:

```python
resultados_pendientes = evaluar(
    RAIZ / "resultados/validacion/preguntas_pendientes.jsonl",
    agente_evaluacion=agente,
)
```

No hace falta repetir las siete preguntas ya respondidas. Esta llamada
guarda otra carpeta y conserva la ejecución anterior. El límite del proveedor
se aplica a solicitudes al modelo, no a preguntas: una pregunta puede
necesitar varias solicitudes. Si vuelve a aparecer un error 429, hay que
esperar el plazo indicado por el proveedor antes de reintentar.
