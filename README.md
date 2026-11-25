# arrow-mlake.mojo

[![mojoshelf](https://mojoshelf.org/badge/arrow-mlake-mojo.svg)](https://mojoshelf.org/tins/arrow-mlake-mojo) [![mojo nightly](https://mojoshelf.org/badge/arrow-mlake-mojo/nightly.svg)](https://mojoshelf.org/tins/arrow-mlake-mojo) [![Used by](https://mojourney.dev/badge/magmalake/arrow-mlake.mojo/used-by.svg)](https://mojourney.dev/repos/magmalake/arrow-mlake.mojo#used-by) [![Mojo version](https://mojourney.dev/badge/magmalake/arrow-mlake.mojo/mojo.svg)](https://mojourney.dev/repos/magmalake/arrow-mlake.mojo)

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
| `arrow_mlake.ipc` | `IpcWriter`, `write_ipc_stream`, `read_ipc_stream` — batches to the Arrow IPC stream format and back |
| `arrow_mlake.flatbuf` | the FlatBuffers builder and bounds-checked reader the IPC metadata is written in |

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
A field's extension name and `PARQUET:field_id` go out as schema metadata, on
nested fields too, so ids survive a round trip.

### One array in

```mojo
var arena = ArrayArena()
var root = import_c(arena, array_addr, schema_addr)          # borrowing
var batch = ImportedArray(array_addr, schema_addr).into_batch()  # owning, no copy
```

`import_c` **borrows**: it reads the pair and copies out of it, and releases
nothing — the caller decides when. `ImportedArray` **owns**: constructing one
asserts "this pair is mine now", and it calls the root release callbacks
exactly once, on `release()` or at destruction. Its `into_batch` and
`into_arena` import without copying and hand the array's release to the
result (see below).

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
batch borrows its array's buffers and releases the array when the batch dies,
so a scan holds as much producer memory as the batches it keeps.

### Arrow IPC streams

The C Data Interface hands buffers to another library in the same process.
The IPC stream format is for everything else: a file, a socket, a shared
mapping another process reads.

```mojo
from arrow_mlake import IpcWriter, read_ipc_stream

var w = IpcWriter(batch)          # the schema comes from this batch
w.write(batch)
var bytes = w^.finish()           # pyarrow's ipc.open_stream reads this

var s = read_ipc_stream(Span(bytes))
print(len(s.batches), "batches;", s.schema.num_columns(), "columns")
```

Every type above goes both ways, nested to any depth, with each field's
nullability, its extension name and its `PARQUET:field_id`. Dictionary
encoding, compression and the IPC *file* format are not implemented, and a
stream that uses them is refused with an error naming the feature. The reader
bounds-checks every FlatBuffers offset and every buffer against the message
body, and copies the buffers into the `ArrayData`, as the C import does.

### Importing without copying

When the release callback is ours (`ImportedArray.into_batch`/`into_arena`,
`ImportedStream.next`), buffers stay in the producer's memory. Each borrowed
`ArrayData` buffer is a capacity-0 `List` over the foreign bytes, and each node
that borrows holds an `ArcPointer[ForeignBuffers]` (`ArrayData.foreign`) that
calls the producer's `release` once, when the last such node is destroyed.

* **Borrowed:** every fixed-width type, utf8/binary (large too), and lists,
  maps and structs of those, sliced or not.
* **Copied anyway:** bool values, a validity bitmap that starts mid-byte, and
  offsets that don't start at zero or aren't aligned (they are rebased; the
  values they index are still borrowed). `import_c` and `import_batch_c` only
  borrow the pair, so they copy everything.
* **Read-only:** writing an element writes the producer's buffer. Pushing to,
  resizing or copying a borrowed list moves it into memory of its own first.
* **Don't move a buffer out:** `node.values^` kept past `node` outlives the
  release. `node.own_buffers()` or `node.copy()` first.

For `struct<int64, utf8>` with short strings, importing costs 0.23 ns per row
without copying, against 6.6 ns per row for the copy in 0.2.1
(`tools/bench_import.mojo`, M4, a million rows).

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
pixi run test                 # Mojo 1.3.0.dev2026112505
pixi run lint                 # mojolint --lsp
pixi run verify-c-import      # the pyarrow gate; needs `uv`
pixi run verify-ipc           # pyarrow and IPC streams, both ways; needs `uv`
```

The unit tests build every array by hand against the columnar spec rather
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

`verify-ipc` does the same for IPC streams, in both directions. pyarrow reads
the test corpus as written here, validates every buffer and writes it back
with its own writer, and that is compared with the corpus. Then pyarrow
writes batches of every supported type, sliced and nested, and this reads
them and writes them again for pyarrow to compare with what it sent.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
