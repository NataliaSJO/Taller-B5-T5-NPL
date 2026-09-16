# Manifiesto del corpus

Generado el 2026-09-02 13:53 UTC por `source/data_source/construir_corpus.py`.

Fuente: SEC EDGAR. Los documentos presentados ante la SEC son
registros públicos y se pueden redistribuir con fines docentes.

## Presentaciones

| Ticker | Empresa | CIK | FY | Cierre | Accession | Origen del FY |
| --- | --- | --- | --- | --- | --- | --- |
| NVDA | NVIDIA CORP | 0001045810 | 2024 | 2024-01-28 | 0001045810-24-000029 | xbrl |
| NVDA | NVIDIA CORP | 0001045810 | 2025 | 2025-01-26 | 0001045810-25-000023 | xbrl |
| MSFT | MICROSOFT CORP | 0000789019 | 2024 | 2024-06-30 | 0000950170-24-087843 | xbrl |
| MSFT | MICROSOFT CORP | 0000789019 | 2025 | 2025-06-30 | 0000950170-25-100235 | xbrl |
| AAPL | Apple Inc. | 0000320193 | 2024 | 2024-09-28 | 0000320193-24-000123 | xbrl |
| AAPL | Apple Inc. | 0000320193 | 2025 | 2025-09-27 | 0000320193-25-000079 | xbrl |
| GOOGL | Alphabet Inc. | 0001652044 | 2024 | 2024-12-31 | 0001652044-25-000014 | xbrl |
| GOOGL | Alphabet Inc. | 0001652044 | 2025 | 2025-12-31 | 0001652044-26-000018 | xbrl |
| META | Meta Platforms, Inc. | 0001326801 | 2024 | 2024-12-31 | 0001326801-25-000017 | xbrl |
| META | Meta Platforms, Inc. | 0001326801 | 2025 | 2025-12-31 | 0001628280-26-003942 | xbrl |
| AMZN | AMAZON COM INC | 0001018724 | 2024 | 2024-12-31 | 0001018724-25-000004 | xbrl |
| AMZN | AMAZON COM INC | 0001018724 | 2025 | 2025-12-31 | 0001018724-26-000004 | xbrl |

## Artefactos

| Fichero | Filas | SHA-256 |
| --- | --- | --- |
| `secciones.jsonl` | 48 | `823272082bcf84fb14f6de415d087f2ee708b77dc475bcf3c3ff1a2d1e487dd1` |
| `chunks.jsonl` | 1749 | `388ff3671742c2248e8f5cb1c75afbc786edfa0dc6f72a62d1fadf2310ec82b2` |
| `xbrl_facts.parquet` | 135 | `f802fc89c2dba96dfef3dd2ef5025e5540620358fc2944e5303093a929127610` |

El hash de `chunks.jsonl` es el que debe citar el manifiesto del
índice FAISS: si no coincide, el índice y sus metadatos están
desalineados (contrato de datos §4).
