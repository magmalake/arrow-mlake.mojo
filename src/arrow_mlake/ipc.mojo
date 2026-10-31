"""The Arrow IPC stream format: `RecordBatch`es to bytes and back.

The C Data Interface hands buffers to another library in the same process.
IPC is for everything else: a file, a socket, a shared mapping another
process reads. pyarrow's `ipc.open_stream` reads what `IpcWriter` writes, and
`read_ipc_stream` reads what pyarrow's `ipc.new_stream` writes.

```mojo
var w = IpcWriter(batch)      # the schema comes from the first batch
w.write(batch)
var bytes = w^.finish()

var s = read_ipc_stream(Span(bytes))
print(len(s.batches), "batches of", s.schema.num_columns(), "columns")
```

## The framing

Each message in a stream is:

    ffffffff                  continuation marker
    <int32 metadata_size>     little-endian, a multiple of 8
    <flatbuffer Message>      padded out to metadata_size
    <body>                    the buffers, each padded to a multiple of 8

and the stream ends with a continuation marker and a zero length. The first
message is the `Schema`; every later one is a `RecordBatch` whose `nodes` and
`buffers` list each array in pre-order, parent before children. The reader
also accepts the pre-0.15 framing without the continuation marker.

## What is supported

Every type `ArrayData` holds: null, bool, the eight integers, the three
floats, utf8 and binary and their large forms, fixed-size binary,
decimal128, date32, time32/64, timestamps with a unit and zone, list, large
list, struct and map, nested to any depth. Fields carry their nullability
and two metadata keys, `ARROW:extension:name` and `PARQUET:field_id`, which
is where `ArrowType.extension` and `ArrayData.field_id` travel.

Dictionary encoding, compression, unions, run-end encoding, the view types
and the IPC *file* format are not implemented; a stream that uses them is
rejected with an error that names the feature.

A validity bitmap is only written when an array has nulls; otherwise the
buffer is empty, which the format allows. Reading copies each buffer into
the `ArrayData`'s own lists.
"""

from std.memory import unsafe_memcpy

from arrow_mlake.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_DECIMAL128,
    AT_FIXED_SIZE_BINARY,
    AT_FLOAT16,
    AT_FLOAT32,
    AT_FLOAT64,
    AT_INT8,
    AT_INT16,
    AT_INT32,
    AT_INT64,
    AT_LARGE_BINARY,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_NULL,
    AT_STRUCT,
    AT_TIME32,
    AT_TIME64,
    AT_TIMESTAMP,
    AT_UINT8,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UTF8,
    TU_MILLI,
    ArrayArena,
    ArrayData,
    ArrowType,
    at,
    at_decimal,
    at_fixed,
    at_time,
    at_timestamp,
    bitmap_bytes,
)
from arrow_mlake.batch import RecordBatch
from arrow_mlake.flatbuf import FlatBufferBuilder, FlatReader

comptime _CONTINUATION = 0xFFFFFFFF

# org.apache.arrow.flatbuf.MetadataVersion
comptime _V4 = 3
comptime _V5 = 4

# MessageHeader union tags
comptime _H_SCHEMA = 1
comptime _H_DICTIONARY_BATCH = 2
comptime _H_RECORD_BATCH = 3

# Type union tags
comptime _T_NULL = 1
comptime _T_INT = 2
comptime _T_FLOATING_POINT = 3
comptime _T_BINARY = 4
comptime _T_UTF8 = 5
comptime _T_BOOL = 6
comptime _T_DECIMAL = 7
comptime _T_DATE = 8
comptime _T_TIME = 9
comptime _T_TIMESTAMP = 10
comptime _T_LIST = 12
comptime _T_STRUCT = 13
comptime _T_FIXED_SIZE_BINARY = 15
comptime _T_MAP = 17
comptime _T_LARGE_BINARY = 19
comptime _T_LARGE_UTF8 = 20
comptime _T_LARGE_LIST = 21

# DateUnit; Date's default is MILLISECOND
comptime _DATE_DAY = 0
comptime _DATE_MILLI = 1

comptime _EXTENSION_NAME = "ARROW:extension:name"
comptime _EXTENSION_METADATA = "ARROW:extension:metadata"
comptime _FIELD_ID = "PARQUET:field_id"


# ── small helpers ──────────────────────────────────────────────────────────


def _pad8(n: Int) -> Int:
    return (n + 7) & ~7


def _append_raw(mut out: List[UInt8], src: Pointer[UInt8, _], n: Int):
    if n <= 0:
        return
    var base = len(out)
    out.resize(unsafe_uninit_length=base + n)
    unsafe_memcpy(dest=out.unsafe_ptr().unsafe_offset(base), src=src, count=n)


def _append_u32(mut out: List[UInt8], v: Int):
    var u = UInt32(v)
    for i in range(4):
        out.append(UInt8((u >> UInt32(i * 8)) & 0xFF))


def _zero_to(mut out: List[UInt8], n: Int):
    while len(out) < n:
        out.append(0)


def _u32_at(data: Span[UInt8, _], pos: Int) -> Int:
    var v = 0
    for i in range(4):
        v |= Int(data[pos + i]) << (8 * i)
    return v


def _is_binary_like(id: Int) -> Bool:
    return (
        id == AT_UTF8
        or id == AT_BINARY
        or id == AT_LARGE_UTF8
        or id == AT_LARGE_BINARY
    )


def _has_offsets(id: Int) -> Bool:
    return (
        _is_binary_like(id)
        or id == AT_LIST
        or id == AT_LARGE_LIST
        or (id == AT_MAP)
    )


def _large_offsets(id: Int) -> Bool:
    return id == AT_LARGE_UTF8 or id == AT_LARGE_BINARY or id == AT_LARGE_LIST


def _children_for(id: Int) -> Int:
    """How many children an array of type `id` has; -1 for any number."""
    if id == AT_LIST or id == AT_LARGE_LIST or id == AT_MAP:
        return 1
    if id == AT_STRUCT:
        return -1
    return 0


# ── writing the schema ─────────────────────────────────────────────────────


def _int_type(id: Int) -> Tuple[Int, Bool]:
    """Bit width and signedness of an integer type, or (0, False)."""
    if id == AT_INT8:
        return (8, True)
    if id == AT_INT16:
        return (16, True)
    if id == AT_INT32:
        return (32, True)
    if id == AT_INT64:
        return (64, True)
    if id == AT_UINT8:
        return (8, False)
    if id == AT_UINT16:
        return (16, False)
    if id == AT_UINT32:
        return (32, False)
    if id == AT_UINT64:
        return (64, False)
    return (0, False)


def _type_tag(t: ArrowType) raises -> UInt8:
    var id = t.id
    if _int_type(id)[0] != 0:
        return _T_INT
    match id:
        case == AT_NULL:
            return _T_NULL
        case == AT_BOOL:
            return _T_BOOL
        case == AT_FLOAT16 | == AT_FLOAT32 | == AT_FLOAT64:
            return _T_FLOATING_POINT
        case == AT_UTF8:
            return _T_UTF8
        case == AT_LARGE_UTF8:
            return _T_LARGE_UTF8
        case == AT_BINARY:
            return _T_BINARY
        case == AT_LARGE_BINARY:
            return _T_LARGE_BINARY
        case == AT_FIXED_SIZE_BINARY:
            return _T_FIXED_SIZE_BINARY
        case == AT_DECIMAL128:
            return _T_DECIMAL
        case == AT_DATE32:
            return _T_DATE
        case == AT_TIME32 | == AT_TIME64:
            return _T_TIME
        case == AT_TIMESTAMP:
            return _T_TIMESTAMP
        case == AT_LIST:
            return _T_LIST
        case == AT_LARGE_LIST:
            return _T_LARGE_LIST
        case == AT_STRUCT:
            return _T_STRUCT
        case == AT_MAP:
            return _T_MAP
        case _:
            raise Error("ipc: cannot write type ", t)


def _write_type(mut b: FlatBufferBuilder, t: ArrowType) raises -> Int:
    """The type table for `t`. Fields equal to the schema's defaults are
    left out, which is how FlatBuffers spells them."""
    var id = t.id
    var int_type = _int_type(id)
    if int_type[0] != 0:
        b.start_table(2)  # Int { bitWidth, is_signed }
        b.add_i32(0, int_type[0])
        b.add_bool(1, int_type[1])
        return b.end_table()
    if id == AT_FLOAT16 or id == AT_FLOAT32 or id == AT_FLOAT64:
        b.start_table(1)  # FloatingPoint { precision: HALF, SINGLE, DOUBLE }
        b.add_i16(0, 0 if id == AT_FLOAT16 else (1 if id == AT_FLOAT32 else 2))
        return b.end_table()
    if id == AT_FIXED_SIZE_BINARY:
        b.start_table(1)  # FixedSizeBinary { byteWidth }
        b.add_i32(0, t.byte_width)
        return b.end_table()
    if id == AT_DECIMAL128:
        b.start_table(3)  # Decimal { precision, scale, bitWidth = 128 }
        b.add_i32(0, t.precision)
        b.add_i32(1, t.scale)
        return b.end_table()
    if id == AT_DATE32:
        b.start_table(1)  # Date { unit = MILLISECOND }
        b.add_i16(0, _DATE_DAY, _DATE_MILLI)
        return b.end_table()
    if id == AT_TIME32 or id == AT_TIME64:
        b.start_table(2)  # Time { unit = MILLISECOND, bitWidth = 32 }
        b.add_i16(0, t.unit, TU_MILLI)
        b.add_i32(1, 32 if id == AT_TIME32 else 64, 32)
        return b.end_table()
    if id == AT_TIMESTAMP:
        # The zone string goes first: offsets only point backwards. An empty
        # zone stays absent, which is how Arrow says "no timezone".
        var tz = 0
        if t.tz:
            tz = b.create_string(t.tz)
        b.start_table(2)  # Timestamp { unit = SECOND, timezone }
        b.add_i16(0, t.unit)
        b.add_offset(1, tz)
        return b.end_table()
    # Null, Bool, Utf8, Binary, their large forms, List, Struct_ and Map
    # (keysSorted = false) are all empty tables.
    _ = _type_tag(t)
    b.start_table(0)
    return b.end_table()


def _write_key_value(
    mut b: FlatBufferBuilder, key: String, value: String
) raises -> Int:
    var k = b.create_string(key)
    var v = b.create_string(value)
    b.start_table(2)  # KeyValue { key, value }
    b.add_offset(0, k)
    b.add_offset(1, v)
    return b.end_table()


def _write_field(
    mut b: FlatBufferBuilder, arena: ArrayArena, node: Int
) raises -> Int:
    """One `Field`, its children first, since a table can only point back."""
    ref a = arena.nodes[node]
    var expect = _children_for(a.type.id)
    if expect >= 0 and len(a.children) != expect:
        raise Error(
            "ipc: ",
            a.type,
            " array ",
            a.name,
            " has ",
            len(a.children),
            " children",
        )
    var children = List[Int]()
    for c in a.children:
        children.append(_write_field(b, arena, c))
    var children_vec = b.create_offset_vector(children)

    var metadata = List[Int]()
    if a.type.extension:
        metadata.append(_write_key_value(b, _EXTENSION_NAME, a.type.extension))
        metadata.append(_write_key_value(b, _EXTENSION_METADATA, String()))
    if a.field_id >= 0:
        metadata.append(_write_key_value(b, _FIELD_ID, String(a.field_id)))
    var metadata_vec = 0
    if len(metadata) > 0:
        metadata_vec = b.create_offset_vector(metadata)

    var name = b.create_string(a.name)
    var type = _write_type(b, a.type)
    # Field { name, nullable, type_type, type, dictionary, children,
    #         custom_metadata }
    b.start_table(7)
    b.add_offset(0, name)
    b.add_bool(1, a.nullable)
    b.add_u8(2, _type_tag(a.type))
    b.add_offset(3, type)
    b.add_offset(5, children_vec)
    b.add_offset(6, metadata_vec)
    return b.end_table()


def _message(
    mut b: FlatBufferBuilder, header_type: UInt8, header: Int, body_length: Int
) raises -> List[UInt8]:
    # Message { version, header_type, header, bodyLength, custom_metadata }
    b.start_table(5)
    b.add_i16(0, _V5)
    b.add_u8(1, header_type)
    b.add_offset(2, header)
    b.add_i64(3, body_length)
    var m = b.end_table()
    return b.finish(m)


def encode_schema_message(batch: RecordBatch) raises -> List[UInt8]:
    """The flatbuffer `Message` holding `batch`'s schema, unframed."""
    var b = FlatBufferBuilder()
    var fields = List[Int]()
    for r in batch.roots:
        fields.append(_write_field(b, batch.arena, r))
    var fields_vec = b.create_offset_vector(fields)
    b.start_table(4)  # Schema { endianness = Little, fields, ... }
    b.add_offset(1, fields_vec)
    var schema = b.end_table()
    return _message(b, _H_SCHEMA, schema, 0)


# ── writing a record batch ─────────────────────────────────────────────────


struct _Body(Movable):
    """A record batch's body as it's built: the bytes, and where each array
    and buffer landed."""

    var bytes: List[UInt8]
    var nodes: List[Int]
    """(length, null_count) pairs, flattened."""
    var buffers: List[Int]
    """(offset, length) pairs, flattened."""

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.nodes = List[Int]()
        self.buffers = List[Int]()

    def add(mut self, src: Pointer[UInt8, _], have: Int, n: Int):
        """Adds a buffer of `n` bytes: the first `have` from `src`, then
        zeros. Bitmaps built bit by bit can end short of their last byte."""
        var at = len(self.bytes)
        _append_raw(self.bytes, src, min(have, n))
        _zero_to(self.bytes, at + n)
        self.buffers.append(at)
        self.buffers.append(n)
        _zero_to(self.bytes, _pad8(len(self.bytes)))

    def add_empty(mut self):
        self.buffers.append(len(self.bytes))
        self.buffers.append(0)


def _write_array(mut body: _Body, arena: ArrayArena, node: Int) raises:
    ref a = arena.nodes[node]
    var id = a.type.id
    var n = a.length
    body.nodes.append(n)
    body.nodes.append(a.null_count)

    if id != AT_NULL:
        if a.null_count > 0:
            if len(a.validity) == 0:
                raise Error(
                    "ipc: array ", a.name, " has nulls but no validity bitmap"
                )
            body.add(a.validity.unsafe_ptr(), len(a.validity), bitmap_bytes(n))
        else:
            body.add_empty()

    if _has_offsets(id):
        var large = _large_offsets(id)
        var have = len(a.large_offsets) if large else len(a.offsets)
        var width = 8 if large else 4
        if have == 0 and n == 0:
            var zero = List[UInt8]()
            _zero_to(zero, width)
            body.add(zero.unsafe_ptr(), width, width)
        elif have < n + 1:
            raise Error(
                "ipc: array ",
                a.name,
                " has ",
                have,
                " offsets for ",
                n,
                " elements",
            )
        elif large:
            body.add(
                a.large_offsets.unsafe_ptr().unsafe_bitcast[UInt8](),
                (n + 1) * 8,
                (n + 1) * 8,
            )
        else:
            body.add(
                a.offsets.unsafe_ptr().unsafe_bitcast[UInt8](),
                (n + 1) * 4,
                (n + 1) * 4,
            )
        if _is_binary_like(id):
            var extent = 0
            if have > 0:
                extent = Int(a.large_offsets[n]) if large else Int(a.offsets[n])
            if len(a.values) < extent:
                raise Error("ipc: array ", a.name, "'s data is short")
            body.add(a.values.unsafe_ptr(), extent, extent)
    elif id == AT_BOOL:
        body.add(a.values.unsafe_ptr(), len(a.values), bitmap_bytes(n))
    elif a.type.fixed_width() > 0:
        var need = n * a.type.fixed_width()
        if len(a.values) < need:
            raise Error(
                "ipc: array ",
                a.name,
                " has ",
                len(a.values),
                " bytes of values, needs ",
                need,
            )
        body.add(a.values.unsafe_ptr(), need, need)

    for c in a.children:
        _write_array(body, arena, c)


def _record_batch_message(length: Int, body: _Body) raises -> List[UInt8]:
    var b = FlatBufferBuilder()
    # `buffers` and `nodes` are vectors of structs, stored inline and back
    # to front.
    var nb = len(body.buffers) // 2
    b.start_vector(16, nb, 8)
    for i in range(nb - 1, -1, -1):
        b.prepend_i64(body.buffers[2 * i + 1])
        b.prepend_i64(body.buffers[2 * i])
    var buffers_vec = b.end_vector()
    var nn = len(body.nodes) // 2
    b.start_vector(16, nn, 8)
    for i in range(nn - 1, -1, -1):
        b.prepend_i64(body.nodes[2 * i + 1])
        b.prepend_i64(body.nodes[2 * i])
    var nodes_vec = b.end_vector()
    # RecordBatch { length, nodes, buffers, compression, variadic... }
    b.start_table(5)
    b.add_i64(0, length)
    b.add_offset(1, nodes_vec)
    b.add_offset(2, buffers_vec)
    var rb = b.end_table()
    return _message(b, _H_RECORD_BATCH, rb, len(body.bytes))


def _frame(mut out: List[UInt8], metadata: List[UInt8], body: List[UInt8]):
    """Appends one framed message. The length written is the padded one, so
    the body, and every buffer in it, starts 8-byte aligned."""
    var start = len(out)
    _append_u32(out, _CONTINUATION)
    _append_u32(out, _pad8(len(metadata)))
    _append_raw(out, metadata.unsafe_ptr(), len(metadata))
    _zero_to(out, start + 8 + _pad8(len(metadata)))
    _append_raw(out, body.unsafe_ptr(), len(body))


struct IpcWriter(Movable):
    """Writes an Arrow IPC stream: the schema, then one message per batch.

    Every batch must have the schema's columns, in order, with the same
    types; `write` checks the top-level types.
    """

    var _out: List[UInt8]
    var _types: List[ArrowType]

    def __init__(out self, schema: RecordBatch) raises:
        """Starts a stream with `schema`'s columns. Only the types and names
        are read; its rows are not written."""
        self._out = List[UInt8]()
        self._types = List[ArrowType]()
        for i in range(schema.num_columns()):
            self._types.append(schema.type(i))
        _frame(self._out, encode_schema_message(schema), List[UInt8]())

    def write(mut self, batch: RecordBatch) raises:
        if batch.num_columns() != len(self._types):
            raise Error(
                "ipc: batch has ",
                batch.num_columns(),
                " columns, schema has ",
                len(self._types),
            )
        var body = _Body()
        for i in range(batch.num_columns()):
            var t = batch.type(i)
            if t.id != self._types[i].id:
                raise Error(
                    "ipc: column ",
                    i,
                    " is ",
                    t,
                    ", schema says ",
                    self._types[i],
                )
            if batch.column(i).length != batch.num_rows:
                raise Error(
                    "ipc: column ",
                    i,
                    " has ",
                    batch.column(i).length,
                    " rows, batch has ",
                    batch.num_rows,
                )
            _write_array(body, batch.arena, batch.roots[i])
        var metadata = _record_batch_message(batch.num_rows, body)
        _frame(self._out, metadata, body.bytes)

    def finish(deinit self) -> List[UInt8]:
        """The finished stream, end-of-stream marker included."""
        var out = self._out^
        _append_u32(out, _CONTINUATION)
        _append_u32(out, 0)
        return out^


def write_ipc_stream(batches: List[RecordBatch]) raises -> List[UInt8]:
    """`batches` as one stream; the first batch gives the schema."""
    if len(batches) == 0:
        raise Error("ipc: a stream needs a schema; pass at least one batch")
    var w = IpcWriter(batches[0])
    for b in batches:
        w.write(b)
    return w^.finish()


# ── reading the schema ─────────────────────────────────────────────────────


struct _Field(Copyable, Movable):
    var name: String
    var nullable: Bool
    var type: ArrowType
    var field_id: Int32
    var children: List[Int]

    def __init__(
        out self, var name: String, nullable: Bool, var type: ArrowType
    ):
        self.name = name^
        self.nullable = nullable
        self.type = type^
        self.field_id = -1
        self.children = List[Int]()


def _read_int(r: FlatReader, t: Int) raises -> ArrowType:
    var width = r.get_i32(t, 0)
    var signed = r.get_bool(t, 1)
    if width == 8:
        return at(AT_INT8 if signed else AT_UINT8)
    if width == 16:
        return at(AT_INT16 if signed else AT_UINT16)
    if width == 32:
        return at(AT_INT32 if signed else AT_UINT32)
    if width == 64:
        return at(AT_INT64 if signed else AT_UINT64)
    raise Error("ipc: integer of bit width ", width)


def _read_type(r: FlatReader, tag: Int, t: Int) raises -> ArrowType:
    """The `ArrowType` for a Type union member. `t` may be 0 for an empty
    table a producer left out."""
    match tag:
        case == _T_NULL:
            return at(AT_NULL)
        case == _T_BOOL:
            return at(AT_BOOL)
        case == _T_UTF8:
            return at(AT_UTF8)
        case == _T_LARGE_UTF8:
            return at(AT_LARGE_UTF8)
        case == _T_BINARY:
            return at(AT_BINARY)
        case == _T_LARGE_BINARY:
            return at(AT_LARGE_BINARY)
        case == _T_LIST:
            return at(AT_LIST)
        case == _T_LARGE_LIST:
            return at(AT_LARGE_LIST)
        case == _T_STRUCT:
            return at(AT_STRUCT)
        case == _T_MAP:
            return at(AT_MAP)
        case _:
            pass
    if t == 0:
        raise Error("ipc: type ", tag, " is missing its parameters")
    match tag:
        case == _T_INT:
            return _read_int(r, t)
        case == _T_FLOATING_POINT:
            var p = r.get_i16(t, 0)
            return at(
                AT_FLOAT16 if p == 0 else (AT_FLOAT32 if p == 1 else AT_FLOAT64)
            )
        case == _T_FIXED_SIZE_BINARY:
            return at_fixed(r.get_i32(t, 0))
        case == _T_DECIMAL:
            var width = r.get_i32(t, 2, 128)
            if width != 128:
                raise Error("ipc: decimal", width, " is not supported")
            return at_decimal(r.get_i32(t, 0), r.get_i32(t, 1))
        case == _T_DATE:
            if r.get_i16(t, 0, _DATE_MILLI) != _DATE_DAY:
                raise Error("ipc: date64 is not supported")
            return at(AT_DATE32)
        case == _T_TIME:
            var unit = r.get_i16(t, 0, TU_MILLI)
            var time = at_time(unit)
            if r.get_i32(t, 1, 32) != (32 if time.id == AT_TIME32 else 64):
                raise Error("ipc: a time's bit width doesn't match its unit")
            return time^
        case == _T_TIMESTAMP:
            return at_timestamp(r.get_i16(t, 0), r.get_string(t, 1))
        case _:
            raise Error("ipc: ", _type_name(tag), " is not supported")


def _type_name(tag: Int) -> String:
    if tag == 11:
        return String("interval")
    if tag == 14:
        return String("union")
    if tag == 16:
        return String("fixed-size list")
    if tag == 18:
        return String("duration")
    if tag == 22:
        return String("run-end encoded")
    if tag >= 23:
        return String("a view type")
    return String("type ", tag)


def _read_field(r: FlatReader, t: Int, mut fields: List[_Field]) raises -> Int:
    """Reads one field and its children into `fields`, parent first."""
    if r.get_ref(t, 4) != 0:
        raise Error("ipc: dictionary-encoded fields are not supported")
    var type = _read_type(r, r.get_u8(t, 2), r.get_ref(t, 3))
    var f = _Field(r.get_string(t, 0), r.get_bool(t, 1), type^)
    var metadata = r.get_ref(t, 6)
    for i in range(r.vector_len(metadata)):
        var kv = r.vector_table(metadata, i)
        var key = r.get_string(kv, 0)
        if key == _EXTENSION_NAME:
            f.type.extension = r.get_string(kv, 1)
        elif key == _FIELD_ID:
            f.field_id = Int32(Int(r.get_string(kv, 1)))
    var index = len(fields)
    var expect = _children_for(f.type.id)
    fields.append(f^)
    var children = r.get_ref(t, 5)
    var n = r.vector_len(children)
    if expect >= 0 and n != expect:
        raise Error(
            "ipc: field ",
            fields[index].name,
            " of type ",
            fields[index].type,
            " has ",
            n,
            " children",
        )
    for i in range(n):
        var c = _read_field(r, r.vector_table(children, i), fields)
        fields[index].children.append(c)
    return index


# ── reading a record batch ─────────────────────────────────────────────────


struct _Cursor:
    """Walks a record batch's `nodes` and `buffers` in step with the
    schema."""

    var nodes: Int
    var buffers: Int
    var node: Int
    var buffer: Int

    def __init__(out self, nodes: Int, buffers: Int):
        self.nodes = nodes
        self.buffers = buffers
        self.node = 0
        self.buffer = 0


def _next_buffer(
    r: FlatReader, mut c: _Cursor, body_len: Int
) raises -> Tuple[Int, Int]:
    if c.buffer >= r.vector_len(c.buffers):
        raise Error("ipc: a record batch has too few buffers")
    var p = r.vector_struct(c.buffers, c.buffer, 16)
    c.buffer += 1
    var off = r.i64(p)
    var n = r.i64(p + 8)
    if off < 0 or n < 0 or off + n > body_len:
        raise Error("ipc: a buffer runs past the message body")
    return (off, n)


def _copy_bytes(
    body: Span[UInt8, _], buf: Tuple[Int, Int], need: Int, what: StringSlice
) raises -> List[UInt8]:
    if buf[1] < need:
        raise Error(
            "ipc: ", what, " buffer has ", buf[1], " bytes, needs ", need
        )
    var out = List[UInt8]()
    _append_raw(out, body.unsafe_ptr().unsafe_offset(buf[0]), need)
    return out^


def _read_array(
    fields: List[_Field],
    f: Int,
    r: FlatReader,
    mut c: _Cursor,
    body: Span[UInt8, _],
    mut arena: ArrayArena,
) raises -> Int:
    ref desc = fields[f]
    if c.node >= r.vector_len(c.nodes):
        raise Error("ipc: a record batch has too few nodes")
    var p = r.vector_struct(c.nodes, c.node, 16)
    c.node += 1
    var a = ArrayData(desc.type.copy(), desc.name.copy())
    a.nullable = desc.nullable
    a.field_id = desc.field_id
    a.length = r.i64(p)
    a.null_count = r.i64(p + 8)
    var n = a.length
    var id = a.type.id
    var body_len = len(body)

    if id != AT_NULL:
        var validity = _next_buffer(r, c, body_len)
        if a.null_count > 0:
            if validity[1] == 0:
                raise Error(
                    "ipc: ", a.name, " has nulls but no validity bitmap"
                )
            a.validity = _copy_bytes(
                body, validity, bitmap_bytes(n), "validity"
            )

    if _has_offsets(id):
        var large = _large_offsets(id)
        var offsets = _next_buffer(r, c, body_len)
        var width = 8 if large else 4
        var extent = 0
        if n == 0 and offsets[1] == 0:
            if large:
                a.large_offsets.append(0)
            else:
                a.offsets.append(0)
        else:
            var bytes = _copy_bytes(body, offsets, (n + 1) * width, "offsets")
            if large:
                a.large_offsets.resize(n + 1, 0)
                unsafe_memcpy(
                    dest=a.large_offsets.unsafe_ptr().unsafe_bitcast[UInt8](),
                    src=bytes.unsafe_ptr(),
                    count=(n + 1) * 8,
                )
                extent = Int(a.large_offsets[n])
            else:
                a.offsets.resize(n + 1, 0)
                unsafe_memcpy(
                    dest=a.offsets.unsafe_ptr().unsafe_bitcast[UInt8](),
                    src=bytes.unsafe_ptr(),
                    count=(n + 1) * 4,
                )
                extent = Int(a.offsets[n])
        if _is_binary_like(id):
            a.values = _copy_bytes(
                body, _next_buffer(r, c, body_len), extent, "data"
            )
    elif id == AT_BOOL:
        a.values = _copy_bytes(
            body, _next_buffer(r, c, body_len), bitmap_bytes(n), "values"
        )
    elif a.type.fixed_width() > 0:
        a.values = _copy_bytes(
            body,
            _next_buffer(r, c, body_len),
            n * a.type.fixed_width(),
            "values",
        )

    var index = arena.add(a^)
    for child in desc.children:
        var ci = _read_array(fields, child, r, c, body, arena)
        arena.nodes[index].children.append(ci)
    return index


def _empty_array(fields: List[_Field], root: Int, mut arena: ArrayArena) -> Int:
    """A zero-length array of field `root`'s type, laid out in pre-order."""
    var stack: List[Int] = [root]
    var parents: List[Int] = [-1]
    var top = -1
    while len(stack):
        var f = stack.pop()
        var parent = parents.pop()
        ref desc = fields[f]
        var a = ArrayData(desc.type.copy(), desc.name.copy())
        a.nullable = desc.nullable
        a.field_id = desc.field_id
        if _has_offsets(a.type.id):
            if _large_offsets(a.type.id):
                a.large_offsets.append(0)
            else:
                a.offsets.append(0)
        var index = arena.add(a^)
        if parent < 0:
            top = index
        else:
            arena.nodes[parent].children.append(index)
        for k in range(len(desc.children) - 1, -1, -1):
            stack.append(desc.children[k])
            parents.append(index)
    return top


struct IpcStream(Movable):
    """What `read_ipc_stream` found: the schema, as a batch of no rows, and
    every record batch in order."""

    var schema: RecordBatch
    var batches: List[RecordBatch]

    def __init__(
        out self, var schema: RecordBatch, var batches: List[RecordBatch]
    ):
        self.schema = schema^
        self.batches = batches^


def read_ipc_stream(data: Span[UInt8, _]) raises -> IpcStream:
    """Reads a whole Arrow IPC stream. Every buffer is copied out of `data`."""
    var fields = List[_Field]()
    var roots = List[Int]()
    var have_schema = False
    var batches = List[RecordBatch]()
    var pos = 0
    while True:
        if pos + 4 > len(data):
            break  # a stream may end without its end-of-stream marker
        var size = _u32_at(data, pos)
        pos += 4
        if size == _CONTINUATION:
            if pos + 4 > len(data):
                raise Error("ipc: stream ends inside a message length")
            size = _u32_at(data, pos)
            pos += 4
        if size == 0:
            break
        if pos + size > len(data):
            raise Error("ipc: stream ends inside a message")
        var r = FlatReader(List[UInt8](data[pos : pos + size]))
        pos += size
        var m = r.root()
        var version = r.get_i16(m, 0)
        if version < _V4:
            raise Error("ipc: metadata version ", version, " is too old")
        var kind = r.get_u8(m, 1)
        var header = r.get_ref(m, 2)
        var body_len = r.get_i64(m, 3)
        if body_len < 0 or pos + body_len > len(data):
            raise Error("ipc: stream ends inside a message body")
        var body = data[pos : pos + body_len]
        pos += body_len
        if header == 0:
            raise Error("ipc: a message has no header")

        if kind == _H_SCHEMA:
            if have_schema:
                raise Error("ipc: a stream has two schemas")
            var fv = r.get_ref(header, 1)
            for i in range(r.vector_len(fv)):
                roots.append(_read_field(r, r.vector_table(fv, i), fields))
            have_schema = True
        elif kind == _H_RECORD_BATCH:
            if not have_schema:
                raise Error("ipc: a record batch comes before the schema")
            if r.get_ref(header, 3) != 0:
                raise Error("ipc: compressed record batches are not supported")
            var c = _Cursor(r.get_ref(header, 1), r.get_ref(header, 2))
            var batch = RecordBatch()
            batch.num_rows = r.get_i64(header, 0)
            for root in roots:
                batch.roots.append(
                    _read_array(fields, root, r, c, body, batch.arena)
                )
            batches.append(batch^)
        elif kind == _H_DICTIONARY_BATCH:
            raise Error("ipc: dictionary batches are not supported")
        else:
            raise Error("ipc: message type ", kind, " is not supported")
    if not have_schema:
        raise Error("ipc: the stream has no schema")
    var schema = RecordBatch()
    for root in roots:
        schema.roots.append(_empty_array(fields, root, schema.arena))
    return IpcStream(schema^, batches^)
