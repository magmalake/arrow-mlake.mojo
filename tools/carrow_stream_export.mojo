"""A shared library that hands pyarrow an `ArrowArrayStream` we built.

The mirror of `carrow_import.mojo`: that one has pyarrow produce and Mojo
consume, this one has Mojo produce and pyarrow consume. Between them both
directions of the C Data Interface are checked against an implementation
nobody here wrote, which is the only way the claim means anything.

Built by the `carrow-stream-lib` task; driven by `tools/verify_c_stream.py`.
"""

from arrow_mlake.arrow import (
    AT_INT64,
    AT_STRUCT,
    ArrayArena,
    ArrayData,
    ArrowType,
)
from arrow_mlake.carrow_stream import export_stream


def _batch(mut arena: ArrayArena, values: List[Int64]) raises -> Int:
    """One batch: `struct<n: int64>`, non-null; returns its root index.

    A record batch is a struct array at the top and one child per column —
    a bare `int64` root is a valid Arrow array but not a valid batch, and a
    consumer will say so.
    """
    var col = ArrayData(ArrowType(AT_INT64), String("n"))
    col.nullable = False
    col.length = len(values)
    col.null_count = 0
    for i in range(len(values)):
        var v = UInt64(values[i])
        for b in range(8):
            col.values.append(UInt8((v >> UInt64(b * 8)) & 0xFF))
    var ci = arena.add(col^)

    var row = ArrayData(ArrowType(AT_STRUCT), String("row"))
    row.nullable = False
    row.length = len(values)
    row.null_count = 0
    row.children = [ci]
    return arena.add(row^)


@export("am_stream_of_three_batches")
def am_stream_of_three_batches() abi("C") -> Int:
    """Three int64 batches — 1..3, 4..5, 6 — as one stream address.

    Uneven on purpose: a consumer that assumed a fixed batch size, or that
    stopped after the first, would still look right on equal batches.
    """
    try:
        var arena = ArrayArena()
        var roots = List[Int]()
        roots.append(_batch(arena, [Int64(1), Int64(2), Int64(3)]))
        roots.append(_batch(arena, [Int64(4), Int64(5)]))
        roots.append(_batch(arena, [Int64(6)]))
        return export_stream(arena, roots)
    except:
        return 0
