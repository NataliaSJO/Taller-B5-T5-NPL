# %% Corpus e indice  --------------------------------
# Los dos ZIP os los pasamos nosotros (Drive compartido, aula virtual o el
# panel de ficheros de Colab): son 5,6 MB entre los dos. Nada de descargar
# de EDGAR en vivo, que con treinta cuadernos a la vez acaba en bloqueo.
#
# Si los teneis en Drive:
#     from google.colab import drive; drive.mount("/content/drive")
# y anadid la carpeta a CANDIDATOS.
import hashlib, pathlib, zipfile

PAQUETES = [
    ("corpus_miax_2026.zip", "4233c37fc9e9d12091af7a146063ad70903a3fe51404a485854f4021c63daee4"),
    ("indice_faiss.zip", "6b5610ad8ac6ea50364445d39bb464d993cbd87048fb07c4fe16657d7ac11655"),
]
URL_RESPALDO = ""          # vacio si no estan alojados
DESTINO = pathlib.Path("corpus")

CANDIDATOS = [
    pathlib.Path("."),
    pathlib.Path("/content"),
    pathlib.Path("/content/drive/MyDrive/MIAX_2026"),
    pathlib.Path("/content/drive/Shareddrives/MIAX_2026"),
]


def _sha256(ruta):
    d = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def _localizar(nombre):
    for base in CANDIDATOS:
        ruta = base / nombre
        if ruta.is_file():
            return ruta
    if URL_RESPALDO:
        import urllib.request
        destino = pathlib.Path(nombre)
        urllib.request.urlretrieve(f"{URL_RESPALDO}/{nombre}", destino)
        return destino
    return None


try:
    for nombre, esperado in PAQUETES:
        origen = _localizar(nombre)
        assert origen is not None, (
            f"No encuentro {nombre}. Subelo con el panel de ficheros de "
            f"Colab (icono de carpeta a la izquierda), o monta el Drive "
            f"donde este. Buscado en: {[str(c) for c in CANDIDATOS]}"
        )
        obtenido = _sha256(origen)
        assert obtenido == esperado, (
            f"{nombre} no coincide con lo esperado: el fichero esta "
            f"corrupto o es de otra version.\n"
            f"  esperado: {esperado}\n  obtenido: {obtenido}"
        )
        with zipfile.ZipFile(origen) as zf:
            zf.extractall(DESTINO)

    # Los dos manifiestos declaran el hash de chunks.jsonl. El indice se
    # construyo sobre ESE fichero: si no cuadra, el indice y sus metadatos
    # estan desalineados y el retrieval devuelve el texto equivocado sin
    # dar ningun error.
    huella = _sha256(DESTINO / "chunks.jsonl")
    for manifiesto in ("MANIFEST.md", "indice/MANIFEST.md"):
        ruta = DESTINO / manifiesto
        if ruta.exists():
            assert huella in ruta.read_text(encoding="utf-8"), (
                f"chunks.jsonl no cuadra con {manifiesto}: el indice se "
                "construyo sobre otros fragmentos."
            )

    print("Corpus e indice verificados en", DESTINO.resolve())
    for p in sorted(DESTINO.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(DESTINO))
            print(f"  {rel:28s} {p.stat().st_size / 1e6:7.2f} MB")

except Exception as e:
    print("No se pudo preparar el corpus:", e)
    print("Pide los ficheros al profesor y dejalos junto al notebook.")
