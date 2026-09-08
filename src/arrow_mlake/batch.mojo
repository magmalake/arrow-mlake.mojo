"""`RecordBatch` — a run of rows as Arrow arrays, one per column.

A batch is `{arena, roots, num_rows}` and nothing else: the arrays live in one
`ArrayArena`, `roots` names the column at the top of each tree, and `num_rows`
is the length every root shares. That is the shape a Parquet reader hands back,
the shape `import_batch_c` unwraps a `+s` root into, and the shape a scan over
the C Data Interface arrives in, so it lives here rather than in any one of
them.

The `column_*` accessors widen a column to a Mojo `List` — `Int64` for any
integer, date, time or timestamp, `Float64` for any float, `String` for utf8
and binary — paired with a per-element validity list. They copy, which is the
point: they are for reading a column out into ordinary Mojo values, not for
the columnar path, where `ArrayData`'s buffers are read in place.

```mojo
var b = import_batch_c(array_addr, schema_addr)
for i in range(b.num_columns()):
    print(b.name(i), String(b.type(i)))
var ids = b.column_i64(0)      # (values, validity)
```
"""

from std.memory import bitcast

from arrow_mlake.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_UINT8,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UTF8,
    ArrayArena,
    ArrayData,
    ArrowType,
    bit_get,
    load_f32,
    load_f64,
)
from arrow_mlake.carrow import ExportedArray, export_c


struct RecordBatch(Copyable, Defaultable, Movable):
    """A contiguous run of rows as Arrow arrays, one per selected column."""

    var arena: ArrayArena
    var roots: List[Int]
    var num_rows: Int

    def __init__(out self):
        self.arena = ArrayArena()
        self.roots = List[Int]()
        self.num_rows = 0

    def __init__(out self, *, copy: Self):
        self.arena = copy.arena.copy()
        self.roots = copy.roots.copy()
        self.num_rows = copy.num_rows

    def __init__(out self, *, deinit move: Self):
        self.arena = move.arena^
        self.roots = move.roots^
        self.num_rows = move.num_rows

    def num_columns(self) -> Int:
        return len(self.roots)

    def column(ref self, i: Int) -> ref[self.arena.nodes[0]] ArrayData:
        return self.arena.nodes[self.roots[i]]

    def child(
        ref self, node: Int, k: Int
    ) -> ref[self.arena.nodes[0]] ArrayData:
        return self.arena.nodes[self.arena.nodes[node].children[k]]

    def name(self, i: Int) -> String:
        return self.arena.nodes[self.roots[i]].name.copy()

    def type(self, i: Int) -> ArrowType:
        return self.arena.nodes[self.roots[i]].type.copy()

    def export_c(self, i: Int) raises -> ExportedArray:
        """Column `i` over the Arrow C Data Interface. The result owns copies
        of every buffer, so it outlives this batch."""
        return export_c(self.arena, self.roots[i])

    def column_i64(self, i: Int) raises -> Tuple[List[Int64], List[Bool]]:
        return array_i64(self.column(i))

    def column_f64(self, i: Int) raises -> Tuple[List[Float64], List[Bool]]:
        return array_f64(self.column(i))

    def column_bool(self, i: Int) raises -> Tuple[List[Bool], List[Bool]]:
        return array_bool(self.column(i))

    def column_str(self, i: Int) raises -> Tuple[List[String], List[Bool]]:
        return array_str(self.column(i))


def _append_validity(a: ArrayData, mut out: List[Bool]):
    for i in range(a.length):
        out.append(bit_get(Span(a.validity), i))


def array_i64_into(
    a: ArrayData, mut vals: List[Int64], mut valid: List[Bool]
) raises:
    """Widen any integer, date, time or timestamp array to `Int64`."""
    var w = a.type.fixed_width()
    if w == 0 or a.type.id == AT_FLOAT32 or a.type.id == AT_FLOAT64:
        raise Error(
            String(
                "arrow_mlake: column of type ",
                String(a.type),
                " is not an integer",
            )
        )
    var signed = not (
        a.type.id == AT_UINT8
        or a.type.id == AT_UINT16
        or a.type.id == AT_UINT32
        or a.type.id == AT_UINT64
    )
    for i in range(a.length):
        var u: UInt64 = 0
        for k in range(w):
            u |= UInt64(a.values[i * w + k]) << UInt64(8 * k)
        if signed and w < 8:
            var sign_bit = UInt64(1) << UInt64(8 * w - 1)
            if (u & sign_bit) != 0:
                u |= ~((UInt64(1) << UInt64(8 * w)) - 1)
        vals.append(bitcast[DType.int64](u))
    _append_validity(a, valid)


def array_f64_into(
    a: ArrayData, mut vals: List[Float64], mut valid: List[Bool]
) raises:
    if a.type.id == AT_FLOAT64:
        for i in range(a.length):
            vals.append(load_f64(Span(a.values), i))
    elif a.type.id == AT_FLOAT32:
        for i in range(a.length):
            vals.append(Float64(load_f32(Span(a.values), i)))
    elif a.type.id == AT_FLOAT16:
        for i in range(a.length):
            var bits = UInt16(a.values[i * 2]) | (
                UInt16(a.values[i * 2 + 1]) << 8
            )
            vals.append(Float64(bitcast[DType.float16](bits)))
    else:
        raise Error(
            String(
                "arrow_mlake: column of type ",
                String(a.type),
                " is not floating point",
            )
        )
    _append_validity(a, valid)


def array_bool_into(
    a: ArrayData, mut vals: List[Bool], mut valid: List[Bool]
) raises:
    if a.type.id != AT_BOOL:
        raise Error(
            String(
                "arrow_mlake: column of type ",
                String(a.type),
                " is not boolean",
            )
        )
    for i in range(a.length):
        vals.append(bit_get(Span(a.values), i))
    _append_validity(a, valid)


def array_str_into(
    a: ArrayData, mut vals: List[String], mut valid: List[Bool]
) raises:
    if a.type.id != AT_UTF8 and a.type.id != AT_BINARY:
        raise Error(
            String(
                "arrow_mlake: column of type ",
                String(a.type),
                " is not a byte array",
            )
        )
    for i in range(a.length):
        var lo = Int(a.offsets[i])
        var hi = Int(a.offsets[i + 1])
        vals.append(String(StringSlice(unsafe_from_utf8=Span(a.values)[lo:hi])))
    _append_validity(a, valid)


def array_i64(a: ArrayData) raises -> Tuple[List[Int64], List[Bool]]:
    var vals = List[Int64]()
    var valid = List[Bool]()
    array_i64_into(a, vals, valid)
    return (vals^, valid^)


def array_f64(a: ArrayData) raises -> Tuple[List[Float64], List[Bool]]:
    var vals = List[Float64]()
    var valid = List[Bool]()
    array_f64_into(a, vals, valid)
    return (vals^, valid^)


def array_bool(a: ArrayData) raises -> Tuple[List[Bool], List[Bool]]:
    var vals = List[Bool]()
    var valid = List[Bool]()
    array_bool_into(a, vals, valid)
    return (vals^, valid^)


def array_str(a: ArrayData) raises -> Tuple[List[String], List[Bool]]:
    var vals = List[String]()
    var valid = List[Bool]()
    array_str_into(a, vals, valid)
    return (vals^, valid^)
