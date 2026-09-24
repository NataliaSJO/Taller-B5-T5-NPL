# Fuentes originales — HTML de EDGAR

Los 12 documentos 10-K de los que salió el corpus, tal cual se
descargaron de SEC EDGAR. Organizados como `TICKER/FYaaaa/documento.htm`.

**No hacen falta para las sesiones.** El corpus ya viene extraído y troceado
en `corpus_miax_2026.zip`. Esto es para quien quiera comprobar la procedencia
de una cita, o intentar la segmentación por Items por su cuenta.

Aviso sobre lo segundo: son HTML con XBRL incrustado, con bloques ocultos y
con la cadena "Item 1A" repetida en el índice, en los encabezados reales, en
las referencias cruzadas y —según el emisor— en el pie de cada página. El
10-K de Microsoft tiene 44 apariciones de "Item 8" a inicio de línea y solo
una es el encabezado. No es una tarde de trabajo.

Procedencia: SEC EDGAR (https://www.sec.gov/edgar). Registros públicos.
Los identificadores exactos están en el `MANIFEST.md` del paquete núcleo.
