# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The initial extraction from
  [parquet.mojo](https://github.com/magmalake/parquet.mojo) 0.8.0.** The Arrow
  memory layout is not Parquet's, and it lived inside `parquet.mojo` only
  because that is where it was first needed. A second consumer — `lancedb.mojo`,
  which imports Arrow batches from a Rust producer over the C Data Interface —
  would have had to take a binding with zero tin dependencies to nine to get at
  it. Four modules moved out unchanged apart from their imports and their error
  prefixes:
  - `arrow_mlake.arrow` — `ArrowType`, `ArrayData`, `ArrayArena`, the `AT_*`
    type ids and the `TU_*` time units, the validity-bitmap helpers
    (`bit_get`, `bit_set`, `bit_fill_valid`, `bitmap_bytes`) and the unaligned
    typed loads (`load_i32` … `load_f64`, `store_u32`, `store_u64`).
  - `arrow_mlake.carrow` — `export_c`, `ExportedArray`, `CArrowSchema`,
    `CArrowArray`: one array out over the C Data Interface, release callbacks
    and all.
  - `arrow_mlake.carrow_import` — `import_c`, `import_batch_c`,
    `ImportedArray`, `ImportedStream`, `parse_format`: the same interface
    inbound, including `ArrowArrayStream`.
  - `arrow_mlake.batch` — `RecordBatch`, which is `{arena, roots, num_rows}`
    and nothing else, with the typed column accessors (`column_i64`,
    `column_f64`, `column_bool`, `column_str`) and the `array_*` kernels behind
    them. It came from `parquet.reader`, where it was the only thing tangling
    the import side back into Parquet.
- **A test suite that builds its own arrays.** The 34 unit tests construct
  every array by hand against the columnar spec rather than reading a file, so
  the coverage is a property of this library rather than of somebody else's
  decoder: no-validity and with-validity, the three offset widths, all four
  nested shapes, the null type with no buffers at all, and an extension type
  whose name has to travel as C metadata. Both negative controls came across —
  a corrupted buffer the round-trip comparator must report, and a permuted
  arena the layout check must catch and the value check must not.
- **The pyarrow gate** (`pixi run verify-c-import`). pyarrow produces 63 cases
  — arrays, a sliced array, and a `RecordBatchReader` stream — this imports
  them and exports them straight back, and pyarrow compares what it gets with
  what it sent. The producer is one we did not write, which is the point: an
  importer checked only against our own exporter proves the two agree and
  nothing else.

### Notes

- **This tin is a stopgap.** See the README: the community Arrow implementation
  for Mojo is [marrow](https://github.com/kszucs/marrow), which is more complete
  than this and which this should be replaced by as soon as it builds on Mojo
  1.x. Read the API as temporary.
