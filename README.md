# arrow-mlake.mojo

[![mojoshelf](https://mojoshelf.org/badge/arrow-mlake-mojo.svg)](https://mojoshelf.org/tins/arrow-mlake-mojo) [![mojo nightly](https://mojoshelf.org/badge/arrow-mlake-mojo/nightly.svg)](https://mojoshelf.org/tins/arrow-mlake-mojo)

[![CI](https://github.com/magmalake/arrow-mlake.mojo/actions/workflows/ci.yml/badge.svg)](https://github.com/magmalake/arrow-mlake.mojo/actions/workflows/ci.yml)

Part of [**magmalake**](https://magmalake.org) — data lake building blocks in Mojo.

The [Apache Arrow](https://arrow.apache.org) memory layout and the [C Data
Interface](https://arrow.apache.org/docs/format/CDataInterface.html), in pure
[Mojo](https://www.modular.com/mojo), with **no tin dependencies and no FFI** —
`std` and nothing else.

## Read this first: it is a stopgap

The community Arrow implementation for Mojo is
[**marrow**](https://github.com/kszucs/marrow), by Krisztián Szűcs. It is more
complete than this will ever be — arrays, buffers, builders, scalars, schema,
expressions, compute kernels, its own C Data Interface and its own Parquet
reader. It currently pins `mojo ==0.26.3.0.dev2026032105`, a pre-1.0 nightly,
and has not been updated since April 2026, so it does not build against Mojo
1.x.

That is the only reason this tin exists. **The intent is to move to marrow as
soon as it builds on Mojo 1.x**, and until then the API here should be read as
temporary: it will change, and it is not trying to be the Arrow library for
Mojo. The name deliberately does not claim `arrow-mojo`, which should stay free
for the community implementation.

What is here is the subset that came out of
[parquet.mojo](https://github.com/magmalake/parquet.mojo), where it lived
because that is where it was first needed. It moved out when a second consumer
appeared: `lancedb.mojo` imports Arrow batches from a Rust producer over the
C Data Interface, and depending on `parquet-mojo` for that would have taken a
binding with **zero** tin dependencies to **nine** — hashes, snappy, zstd, lz4,
thrift, avro, threads, brotli — including three compression codecs and a
Thrift implementation it would never execute. `parquet-mojo` now depends on this and re-exports it, so
`from parquet.arrow import ArrayData` still resolves.

## What it does

| module | what is in it |
|---|---|
| `arrow_mlake.arrow` | `ArrowType`, `ArrayData`, `ArrayArena`, the `AT_*` type ids, validity-bitmap helpers, unaligned typed loads |
| `arrow_mlake.carrow` | `export_c` — an array out over the C Data Interface, release callbacks and all |
| `arrow_mlake.carrow_import` | `import_c`, `import_batch_c`, `ImportedArray`, `ImportedStream` — the same interface inbound, including `ArrowArrayStream` |
| `arrow_mlake.batch` | `RecordBatch`, a run of rows as one array per column, with typed accessors |

Types it can name, both directions: `null`, `bool`, every signed and unsigned
integer width, `float16`/`float32`/`float64`, `utf8` and `large_utf8`, `binary`
and `large_binary`, `fixed_size_binary`, `decimal128`, `date32`, `time32`,
`time64`, `timestamp` (with time zone), and the four nested shapes — `list`,
`large_list`, `struct` and `map` — plus Arrow extension types, whose
`ARROW:extension:name` travels as C metadata. Anything else is an **error
carrying the format string**, never a guess: unions, run-end encoding, list
views, fixed-size lists, `date64`, durations, intervals, decimal32/64/256 and
the view types are named and refused. Guessing at an unrecognised format would
mean reinterpreting somebody else's buffer, which is the one failure mode that
produces wrong numbers silently rather than an error.

Dictionary-encoded arrays are refused as well; there is no dictionary in
`ArrayData` to put one in.

## API

```mojo
from arrow_mlake import (
    ArrayArena, ArrayData, ArrowType, RecordBatch,
    ImportedArray, ImportedStream,
    export_c, import_c, import_batch_c, parse_format,
)
```

### One array out

```mojo
var e = export_c(arena, root)
print(e.array, e.schema)   # two addresses, ready for _import_from_c
var raw = e.into_raw()     # hand ownership to the consumer
```

Each export is **two** allocations — one whole schema tree, one whole array
tree — with the buffers copied rather than borrowed, so the result outlives the
arena it came from and is safe to hand to a runtime that will release it
whenever it likes. The root's `release` frees the lot; children carry a
callback that only clears themselves, because the interface says a consumer
releases the root and the root owns everything. Both callbacks are `abi("C")`.

### One array in

```mojo
var arena = ArrayArena()
var root = import_c(arena, array_addr, schema_addr)          # borrowing
var batch = ImportedArray(array_addr, schema_addr).into_batch()  # owning
```

`import_c` **borrows**: it reads the pair and copies out of it, and releases
nothing — the caller decides when. `ImportedArray` **owns**: constructing one
asserts "this pair is mine now", and it calls the root release callbacks
exactly once, on `release()` or at destruction.

Nodes land in the arena in DFS **pre-order**, which is the order the exporter
walks on the way out, so `export → import → export` reproduces the arena's
shape and not merely its numbers.

### A stream of batches

```mojo
var s = ImportedStream(stream_addr)   # takes ownership; pulls the schema
while True:
    var got = s.next()
    if not got:
        break
    ref batch = got.value()
    print(batch.num_rows, "rows")
```

`ArrowArrayStream` is what a scan actually returns, so it is the entry point a
LanceDB or ADBC binding needs. The end of the stream is a *released* array,
which is the only signal the interface has for "no more"; `get_next` returning
non-zero is a real error and `get_last_error`'s message is carried into the
raise, because conflating the two turns a failed scan into a short one. Each
batch is released as soon as it has been copied, so a long scan holds one
batch of producer memory at a time rather than the whole stream.

### Importing copies

`ArrayData` owns its buffers as `List[UInt8]`, so every buffer a producer hands
over is copied into Mojo memory. The zero-copy alternative — an `ArrayData`
that borrows a foreign pointer and keeps the producer's release callback alive
behind it — changes the type every consumer reads, and is deliberately not what
this does. On a 16-column, 65k-row batch the copy runs at memory-copy speed.

### Nothing about a producer is trusted

`n_buffers` is checked against the format string, `n_children` against the
type, offsets for monotonicity and against the child length they index into,
nesting depth against `MAX_IMPORT_DEPTH` (a cycle in a producer's pointers
would otherwise be an unbounded walk), and the metadata block's own counts
before they are used as loop bounds. The interface carries no buffer sizes — a
`const void**` and nothing more — so a truncated `utf8` data buffer is
undetectable by construction; what can be cross-checked is checked, and the
limit is stated rather than papered over.

A producer's `offset` is honoured by materialising the window it names —
validity bitmaps re-packed bit by bit when the start is unaligned, values,
offsets, and recursively the children a list's offsets point into. `ArrayData`
has no offset field, so that is the only faithful reading of a sliced array.

## Install as a mojoshelf tin

```sh
pixi shelf add arrow-mlake-mojo     # pixi mode (git source dependency)
```

Working with a coding agent? `npx skills add mojoshelf/mojoshelf --skill mojoshelf-consume --yes` teaches it to find and install tins itself — it installs the `shelf` CLI too.

Or as a plain source dependency: `-I ../arrow-mlake.mojo/src`, no FFI, no link
flags, nothing else to check out.

## Test

```sh
pixi run -e stable test       # stable Mojo 1.0.0
pixi run -e default test      # nightly
pixi run -e default lint      # mojolint --lsp
pixi run verify-c-import      # the pyarrow gate; needs `uv`
```

The 34 unit tests build every array by hand against the columnar spec rather
than reading a file, so the coverage is a property of this library and not of
somebody else's decoder: no-validity and with-validity, the three offset
widths, all four nested shapes, the null type with no buffers at all, and an
extension type that has to carry its metadata through C. Both negative controls
are there too — a corrupted buffer that the round-trip comparator must report,
and a permuted arena that the layout check must catch and the value check must
not.

`verify-c-import` is the cross-language gate, and the one that matters: pyarrow
*produces* 63 cases — arrays, a sliced array, a `RecordBatchReader` stream —
`_export_to_c` moves them over, and this imports them and exports them straight
back for pyarrow to compare with what it sent. An importer checked only against
our own exporter would prove the two agree and nothing else.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
