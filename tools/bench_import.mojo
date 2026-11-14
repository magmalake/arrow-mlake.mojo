"""Per-row cost of importing a batch over the C Data Interface.

The batch is the lib-tin spike's (morca #454): `struct<id: int64, name: utf8>`
with short strings, exported once per iteration by `export_c`. Only the
import and the release are timed — the part a scan pays on the consumer's
side. `copy` is `import_batch_c` (borrows the pair, copies every buffer);
`zero-copy` is `ImportedArray.into_batch`, which keeps the producer's buffers
and releases them when the batch dies.

```console
mojo build tools/bench_import.mojo -I src -I ../memory-region.mojo/src -o build/bench-import
./build/bench-import
```
"""

from std.time import perf_counter_ns

from arrow_mlake import (
    AT_INT64,
    AT_STRUCT,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    ArrowType,
    ImportedArray,
    export_c,
    import_batch_c,
    store_u64,
)


def _batch(rows: Int) -> Tuple[ArrayArena, Int]:
    var arena = ArrayArena()
    var id = ArrayData(ArrowType(AT_INT64), String("id"))
    id.length = rows
    var name = ArrayData(ArrowType(AT_UTF8), String("name"))
    name.length = rows
    name.offsets.append(0)
    for i in range(rows):
        store_u64(id.values, UInt64(i))
        var s = String("user-", i % 1000)
        name.values.extend(s.as_bytes())
        name.offsets.append(Int32(len(name.values)))
    var ci = arena.add(id^)
    var cn = arena.add(name^)
    var row = ArrayData(ArrowType(AT_STRUCT), String("row"))
    row.length = rows
    row.children = [ci, cn]
    var r = arena.add(row^)
    return (arena^, r)


def _run(rows: Int, zero_copy: Bool, iters: Int) raises -> Float64:
    var b = _batch(rows)
    var best = Float64(1e30)
    var checksum = 0
    for _ in range(iters):
        var e = export_c(b[0], b[1])
        var raw = e.into_raw()
        var t0 = perf_counter_ns()
        if zero_copy:
            var batch = ImportedArray(raw[0], raw[1]).into_batch()
            checksum += batch.num_rows
        else:
            var batch = import_batch_c(raw[0], raw[1])
            checksum += batch.num_rows
            var owned = ImportedArray(raw[0], raw[1])
            owned.release()
        var t1 = perf_counter_ns()
        best = min(best, Float64(t1 - t0) / Float64(rows))
    if checksum != rows * iters:
        raise Error("bench: lost rows")
    return best


def main() raises:
    var sizes: List[Int] = [1000, 100_000, 1_000_000]
    for i in range(len(sizes)):
        var n = sizes[i]
        var iters = 50 if n < 1_000_000 else 10
        print(
            n,
            "rows  copy:",
            _run(n, False, iters),
            "ns/row  zero-copy:",
            _run(n, True, iters),
            "ns/row",
        )
