# Corpus 10-K — prácticas MIAX 2026

Informes anuales 10-K presentados ante la SEC.
Empresas: NVDA, MSFT, AAPL, GOOGL, META, AMZN. Ejercicios: FY2024 y FY2025.
Empaquetado el 2026-09-02.

## Qué hay aquí

| Fichero | Qué es | Herramienta |
| --- | --- | --- |
| `secciones.jsonl` | Texto íntegro de cada sección | `read_section` |
| `chunks.jsonl` | Ese mismo texto, troceado | `search_filings` |
| `xbrl_facts.parquet` | Las cifras reportadas | `get_xbrl_fact` |
| `MANIFEST.md` | Procedencia y hashes SHA-256 | — |

`secciones.jsonl` lleva una línea por (empresa, ejercicio, item).
`chunks.jsonl`, fragmentos de ~500 tokens con solape de 80, cada uno con
su `chunk_id`, que es lo que permite verificar una cita.

Items incluidos: 1A, 7, 7A, 8.

## Tres cosas que conviene saber antes de usarlo

**`fiscal_year` no es el año de presentación.** NVIDIA cierra ejercicio en
enero: su FY2025 acabó el 26 de enero de 2025 y el informe se presentó en
febrero de 2025. Alphabet cierra en diciembre: su FY2025 se presentó en
febrero de 2026. Fíate del campo, no de la fecha.

**El tabulador solo aparece dentro de tablas.** Las celdas de una fila van
unidas por `\t` y las filas por `\n`; en el texto corrido no hay tabuladores.
Por eso `contiene_tabla` es exacto y no una estimación.

**Las cifras salen de XBRL, no del texto.** `xbrl_facts.parquet` es la fuente
autorizada. Un número leído del texto de una tabla troceada no lo es.

## Procedencia y licencia

Todo procede de SEC EDGAR (https://www.sec.gov/edgar). Los informes 10-K son
registros públicos presentados ante la Securities and Exchange Commission de
EE. UU. El `MANIFEST.md` lleva el CIK, el número de accession y la fecha de
cierre de cada presentación, y cada fila de `secciones.jsonl` lleva su
`url_origen`: cualquier cita se puede verificar contra el documento original.
