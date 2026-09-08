"""Arrow arrays built by hand, so the tests need no file format to make data.

`parquet.mojo` used to test this code by reading a fixture and exporting what
came out, which made the Arrow layer's coverage a function of somebody else's
decoder. Here the arrays are written a byte at a time against the columnar
spec, which is what the export and the import are being checked against
anyway.

`sample_arena` is the corpus: one arena of sixteen top-level columns covering
every buffer layout this library knows — no-validity and with-validity, the
three offset widths, the four nested shapes, a null array with no buffers at
all, and an extension type that has to carry its metadata through C.
"""

from std.memory import bitcast

from arrow_mlake import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_INT64,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_NULL,
    AT_STRUCT,
    AT_UTF8,
    AT_FLOAT64,
    TU_MICRO,
    ArrayArena,
    ArrayData,
    ArrowType,
    at_decimal,
    at_fixed,
    at_timestamp,
    bit_set,
    store_u64,
)


def put_i64(mut buf: List[UInt8], v: Int64):
    store_u64(buf, UInt64(v))


def put_f64(mut buf: List[UInt8], v: Float64):
    store_u64(buf, bitcast[DType.uint64](v))


def put_i32(mut buf: List[UInt8], v: Int32):
    var u = UInt32(v)
    for k in range(4):
        buf.append(UInt8((u >> UInt32(8 * k)) & 0xFF))


def _null_at(i: Int, every: Int) -> Bool:
    """Every `every`-th element is null; `every <= 0` means none are."""
    return every > 0 and i % every == every - 1


# ── flat arrays ────────────────────────────────────────────────────────────


def int64_array(
    var name: String, rows: Int, base: Int64, every: Int
) raises -> ArrayData:
    """`int64` holding `base + i`, with every `every`-th element null."""
    var a = ArrayData(ArrowType(AT_INT64), name^)
    a.length = rows
    for i in range(rows):
        var null = _null_at(i, every)
        put_i64(a.values, 0 if null else base + Int64(i))
        if null:
            a.null_count += 1
        if every > 0:
            bit_set(a.validity, i, not null)
    return a^


def float64_array(var name: String, rows: Int) raises -> ArrayData:
    var a = ArrayData(ArrowType(AT_FLOAT64), name^)
    a.length = rows
    for i in range(rows):
        put_f64(a.values, Float64(i) + 0.5)
    return a^


def bool_array(var name: String, rows: Int) raises -> ArrayData:
    """`bool`, values bit-packed like the validity bitmap beside them."""
    var a = ArrayData(ArrowType(AT_BOOL), name^)
    a.length = rows
    for i in range(rows):
        var null = _null_at(i, 5)
        bit_set(a.values, i, (i % 2) == 0 and not null)
        bit_set(a.validity, i, not null)
        if null:
            a.null_count += 1
    return a^


def utf8_array(var name: String, rows: Int, large: Bool) raises -> ArrayData:
    """`utf8` or `large_utf8`; element 0 is the empty string, every 4th null."""
    var a = ArrayData(ArrowType(AT_LARGE_UTF8 if large else AT_UTF8), name^)
    a.length = rows
    if large:
        a.large_offsets.append(0)
    else:
        a.offsets.append(0)
    for i in range(rows):
        var null = _null_at(i, 4)
        if not null and i > 0:
            var s = String("value-", i)
            a.values.extend(s.as_bytes())
        if null:
            a.null_count += 1
        bit_set(a.validity, i, not null)
        if large:
            a.large_offsets.append(Int64(len(a.values)))
        else:
            a.offsets.append(Int32(len(a.values)))
    return a^


def binary_array(var name: String, rows: Int) raises -> ArrayData:
    """`binary`, holding bytes that are deliberately not valid UTF-8."""
    var a = ArrayData(ArrowType(AT_BINARY), name^)
    a.length = rows
    a.offsets.append(0)
    for i in range(rows):
        for k in range(i % 5):
            a.values.append(UInt8(0x80 + k))
        a.offsets.append(Int32(len(a.values)))
    return a^


def decimal_array(var name: String, rows: Int) raises -> ArrayData:
    """`decimal128(38, 9)` — sixteen little-endian bytes per element."""
    var a = ArrayData(at_decimal(38, 9), name^)
    a.length = rows
    for i in range(rows):
        put_i64(a.values, Int64(1000 + i))
        put_i64(a.values, 0)
    return a^


def uuid_array(var name: String, rows: Int) raises -> ArrayData:
    """`fixed_size_binary(16)` carrying the `arrow.uuid` extension name.

    The extension name is the only thing in `ArrowType` that travels as C
    *metadata* rather than as a format string, so it is the one field a round
    trip can lose silently.
    """
    var t = at_fixed(16)
    t.extension = String("arrow.uuid")
    var a = ArrayData(t^, name^)
    a.length = rows
    for i in range(rows):
        for k in range(16):
            a.values.append(UInt8((i * 16 + k) % 256))
    return a^


def timestamp_array(var name: String, rows: Int) raises -> ArrayData:
    var a = ArrayData(at_timestamp(TU_MICRO, String("UTC")), name^)
    a.length = rows
    for i in range(rows):
        put_i64(a.values, Int64(1510871468000000) + Int64(i))
    return a^


def date32_array(var name: String, rows: Int) raises -> ArrayData:
    var a = ArrayData(ArrowType(AT_DATE32), name^)
    a.length = rows
    for i in range(rows):
        put_i32(a.values, Int32(17486 + i))
    return a^


def null_array(var name: String, rows: Int) raises -> ArrayData:
    """The `null` type: no buffers at all, every element null by construction.
    """
    var a = ArrayData(ArrowType(AT_NULL), name^)
    a.length = rows
    a.null_count = rows
    return a^


# ── nested arrays ──────────────────────────────────────────────────────────


def add_list(
    mut arena: ArrayArena, var name: String, rows: Int, large: Bool
) raises -> Int:
    """`list<int64>` (or `large_list`) of `rows` cells; every third is null.

    A null cell is an empty run, which is what makes the offsets worth
    checking: a consumer that ignored validity would read the same values.
    """
    var lengths = List[Int]()
    var total = 0
    for i in range(rows):
        var n = 0 if _null_at(i, 3) else (i % 4)
        lengths.append(n)
        total += n
    var child = int64_array(String("item"), total, 1000, 0)
    var ci = arena.add(child^)
    var a = ArrayData(ArrowType(AT_LARGE_LIST if large else AT_LIST), name^)
    a.length = rows
    a.children = [ci]
    var at = 0
    if large:
        a.large_offsets.append(0)
    else:
        a.offsets.append(0)
    for i in range(rows):
        var null = _null_at(i, 3)
        at += lengths[i]
        if null:
            a.null_count += 1
        bit_set(a.validity, i, not null)
        if large:
            a.large_offsets.append(Int64(at))
        else:
            a.offsets.append(Int32(at))
    return arena.add(a^)


def add_struct(
    mut arena: ArrayArena, var name: String, rows: Int
) raises -> Int:
    """`struct<a: int64, b: utf8>`, with nulls inside the fields and not on it.
    """
    var ai = arena.add(int64_array(String("a"), rows, 7000, 3))
    var bi = arena.add(utf8_array(String("b"), rows, False))
    var a = ArrayData(ArrowType(AT_STRUCT), name^)
    a.length = rows
    a.children = [ai, bi]
    return arena.add(a^)


def add_map(mut arena: ArrayArena, var name: String, rows: Int) raises -> Int:
    """`map<utf8, int64>` — a list of a two-field struct, per the Arrow spec.

    The key field is not nullable, which is a flag the C schema carries and a
    round trip can therefore drop.
    """
    var entries = 0
    var lengths = List[Int]()
    for i in range(rows):
        var n = i % 3
        lengths.append(n)
        entries += n
    var key = utf8_array(String("key"), entries, False)
    key.nullable = False
    key.null_count = 0
    key.validity = List[UInt8]()
    var ki = arena.add(key^)
    var vi = arena.add(int64_array(String("value"), entries, 500, 0))
    var kv = ArrayData(ArrowType(AT_STRUCT), String("entries"))
    kv.nullable = False
    kv.length = entries
    kv.children = [ki, vi]
    var kvi = arena.add(kv^)
    var a = ArrayData(ArrowType(AT_MAP), name^)
    a.length = rows
    a.children = [kvi]
    a.offsets.append(0)
    var at = 0
    for i in range(rows):
        at += lengths[i]
        a.offsets.append(Int32(at))
    return arena.add(a^)


# ── the corpus ─────────────────────────────────────────────────────────────

comptime SAMPLE_ROWS = 11


def sample_arena() raises -> Tuple[ArrayArena, List[Int]]:
    """One arena, sixteen top-level columns, every buffer layout we can write.
    """
    var arena = ArrayArena()
    var roots = List[Int]()
    roots.append(arena.add(int64_array(String("i64"), SAMPLE_ROWS, -5, 3)))
    roots.append(arena.add(int64_array(String("plain"), SAMPLE_ROWS, 42, 0)))
    roots.append(arena.add(float64_array(String("f64"), SAMPLE_ROWS)))
    roots.append(arena.add(bool_array(String("b"), SAMPLE_ROWS)))
    roots.append(arena.add(utf8_array(String("s"), SAMPLE_ROWS, False)))
    roots.append(arena.add(utf8_array(String("ls"), SAMPLE_ROWS, True)))
    roots.append(arena.add(binary_array(String("bin"), SAMPLE_ROWS)))
    roots.append(arena.add(decimal_array(String("dec"), SAMPLE_ROWS)))
    roots.append(arena.add(uuid_array(String("uuid"), SAMPLE_ROWS)))
    roots.append(arena.add(timestamp_array(String("ts"), SAMPLE_ROWS)))
    roots.append(arena.add(date32_array(String("d"), SAMPLE_ROWS)))
    roots.append(arena.add(null_array(String("nul"), SAMPLE_ROWS)))
    roots.append(add_list(arena, String("li"), SAMPLE_ROWS, False))
    roots.append(add_list(arena, String("ll"), SAMPLE_ROWS, True))
    roots.append(add_struct(arena, String("st"), SAMPLE_ROWS))
    roots.append(add_map(arena, String("m"), SAMPLE_ROWS))
    return (arena^, roots^)
