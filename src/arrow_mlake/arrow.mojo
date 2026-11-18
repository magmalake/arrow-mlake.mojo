"""Arrow types and Arrow array data, in the layout the C Data Interface wants.

`ArrayData` is one Arrow array: a type, a length, an optional validity bitmap,
up to one offsets buffer, a values buffer, and children. Nesting is stored in
an *arena* (`ArrayArena`) with children referenced by index, because Mojo does
not allow a struct to contain a `List` of itself.

Buffer layouts follow the Arrow columnar spec exactly, so
`arrow_mlake.carrow` can hand the buffers to another runtime without touching
them:

| type | buffers |
|---|---|
| null | — |
| boolean | validity, values (bit-packed, LSB first) |
| fixed width | validity, values |
| utf8 / binary | validity, offsets (i32), data |
| large utf8 / large binary | validity, offsets (i64), data |
| fixed-size binary, decimal128 | validity, values |
| list / map | validity, offsets (i32) + one child |
| large list | validity, offsets (i64) + one child |
| struct | validity + N children |
"""

from std.memory import ArcPointer
from std.os import abort

# ── Arrow type ids ─────────────────────────────────────────────────────────

comptime AT_NULL = 0
comptime AT_BOOL = 1
comptime AT_INT8 = 2
comptime AT_INT16 = 3
comptime AT_INT32 = 4
comptime AT_INT64 = 5
comptime AT_UINT8 = 6
comptime AT_UINT16 = 7
comptime AT_UINT32 = 8
comptime AT_UINT64 = 9
comptime AT_FLOAT16 = 10
comptime AT_FLOAT32 = 11
comptime AT_FLOAT64 = 12
comptime AT_UTF8 = 13
comptime AT_LARGE_UTF8 = 14
comptime AT_BINARY = 15
comptime AT_LARGE_BINARY = 16
comptime AT_FIXED_SIZE_BINARY = 17
comptime AT_DECIMAL128 = 18
comptime AT_DATE32 = 19
comptime AT_TIME32 = 20
comptime AT_TIME64 = 21
comptime AT_TIMESTAMP = 22
comptime AT_LIST = 23
comptime AT_LARGE_LIST = 24
comptime AT_STRUCT = 25
comptime AT_MAP = 26

# Time units.
comptime TU_SECOND = 0
comptime TU_MILLI = 1
comptime TU_MICRO = 2
comptime TU_NANO = 3


def unit_suffix(unit: Int) -> String:
    if unit == TU_SECOND:
        return String("s")
    if unit == TU_MILLI:
        return String("m")
    if unit == TU_MICRO:
        return String("u")
    return String("n")


def unit_name(unit: Int) -> String:
    if unit == TU_SECOND:
        return String("s")
    if unit == TU_MILLI:
        return String("ms")
    if unit == TU_MICRO:
        return String("us")
    return String("ns")


@fieldwise_init
struct ArrowType(Copyable, Movable, Writable):
    """One Arrow logical type. `id` is one of the `AT_*` constants."""

    var id: Int
    var byte_width: Int
    """Element width for `AT_FIXED_SIZE_BINARY` and `AT_DECIMAL128` (16)."""
    var precision: Int
    var scale: Int
    var unit: Int
    """Time unit for time/timestamp types, one of the `TU_*` constants."""
    var tz: String
    """`"UTC"` for a UTC-adjusted timestamp, empty for a local one."""
    var extension: String
    """An Arrow extension name such as `arrow.uuid`, or empty."""

    def __init__(out self, id: Int):
        self.id = id
        self.byte_width = 0
        self.precision = 0
        self.scale = 0
        self.unit = TU_MICRO
        self.tz = String()
        self.extension = String()

    def __init__(out self):
        self = ArrowType(AT_NULL)

    def is_nested(self) -> Bool:
        return (
            self.id == AT_LIST
            or self.id == AT_LARGE_LIST
            or self.id == AT_STRUCT
            or self.id == AT_MAP
        )

    def fixed_width(self) -> Int:
        """Bytes per element for a fixed-width type, or 0 if not fixed width."""
        if self.id == AT_INT8 or self.id == AT_UINT8:
            return 1
        if self.id == AT_INT16 or self.id == AT_UINT16 or self.id == AT_FLOAT16:
            return 2
        if (
            self.id == AT_INT32
            or self.id == AT_UINT32
            or self.id == AT_FLOAT32
            or self.id == AT_DATE32
            or self.id == AT_TIME32
        ):
            return 4
        if (
            self.id == AT_INT64
            or self.id == AT_UINT64
            or self.id == AT_FLOAT64
            or self.id == AT_TIME64
            or self.id == AT_TIMESTAMP
        ):
            return 8
        if self.id == AT_DECIMAL128:
            return 16
        if self.id == AT_FIXED_SIZE_BINARY:
            return self.byte_width
        return 0

    def format(self) -> String:
        """The Arrow C Data Interface format string for this type."""
        var i = self.id
        match i:
            case == AT_NULL:
                return String("n")
            case == AT_BOOL:
                return String("b")
            case == AT_INT8:
                return String("c")
            case == AT_UINT8:
                return String("C")
            case == AT_INT16:
                return String("s")
            case == AT_UINT16:
                return String("S")
            case == AT_INT32:
                return String("i")
            case == AT_UINT32:
                return String("I")
            case == AT_INT64:
                return String("l")
            case == AT_UINT64:
                return String("L")
            case == AT_FLOAT16:
                return String("e")
            case == AT_FLOAT32:
                return String("f")
            case == AT_FLOAT64:
                return String("g")
            case == AT_UTF8:
                return String("u")
            case == AT_LARGE_UTF8:
                return String("U")
            case == AT_BINARY:
                return String("z")
            case == AT_LARGE_BINARY:
                return String("Z")
            case == AT_FIXED_SIZE_BINARY:
                return String("w:", self.byte_width)
            case == AT_DECIMAL128:
                return String("d:", self.precision, ",", self.scale)
            case == AT_DATE32:
                return String("tdD")
            case == AT_TIME32 | == AT_TIME64:
                return String("tt", unit_suffix(self.unit))
            case == AT_TIMESTAMP:
                return String("ts", unit_suffix(self.unit), ":", self.tz)
            case == AT_LIST:
                return String("+l")
            case == AT_LARGE_LIST:
                return String("+L")
            case == AT_STRUCT:
                return String("+s")
            case == AT_MAP:
                return String("+m")
            case _:
                return String("n")

    def write_to(self, mut writer: Some[Writer]):
        var i = self.id
        match i:
            case == AT_DECIMAL128:
                writer.write(
                    "decimal128(", self.precision, ", ", self.scale, ")"
                )
            case == AT_FIXED_SIZE_BINARY:
                writer.write("fixed_size_binary[", self.byte_width, "]")
            case == AT_TIMESTAMP:
                writer.write("timestamp[", unit_name(self.unit))
                if self.tz:
                    writer.write(", tz=", self.tz)
                writer.write("]")
            case == AT_TIME32 | == AT_TIME64:
                writer.write("time", 32 if i == AT_TIME32 else 64)
                writer.write("[", unit_name(self.unit), "]")
            case _:
                writer.write(_type_name(i))
        if self.extension:
            writer.write(" <", self.extension, ">")


def _type_name(i: Int) -> String:
    var names: List[String] = [
        String("null"),
        String("bool"),
        String("int8"),
        String("int16"),
        String("int32"),
        String("int64"),
        String("uint8"),
        String("uint16"),
        String("uint32"),
        String("uint64"),
        String("halffloat"),
        String("float"),
        String("double"),
        String("string"),
        String("large_string"),
        String("binary"),
        String("large_binary"),
        String("fixed_size_binary"),
        String("decimal128"),
        String("date32[day]"),
        String("time32"),
        String("time64"),
        String("timestamp"),
        String("list"),
        String("large_list"),
        String("struct"),
        String("map"),
    ]
    if i < 0 or i >= len(names):
        return String("?")
    return names[i].copy()


# ── constructors for the common types ──────────────────────────────────────


def at(id: Int) -> ArrowType:
    return ArrowType(id)


def at_fixed(width: Int) -> ArrowType:
    var t = ArrowType(AT_FIXED_SIZE_BINARY)
    t.byte_width = width
    return t^


def at_decimal(precision: Int, scale: Int) -> ArrowType:
    var t = ArrowType(AT_DECIMAL128)
    t.byte_width = 16
    t.precision = precision
    t.scale = scale
    return t^


def at_timestamp(unit: Int, var tz: String) -> ArrowType:
    var t = ArrowType(AT_TIMESTAMP)
    t.unit = unit
    t.tz = tz^
    return t^


def at_time(unit: Int) -> ArrowType:
    var t = ArrowType(AT_TIME32 if unit <= TU_MILLI else AT_TIME64)
    t.unit = unit
    return t^


# ── validity bitmaps ───────────────────────────────────────────────────────


def bitmap_bytes(n: Int) -> Int:
    return (n + 7) // 8


def bit_get(bitmap: Span[UInt8, _], i: Int) -> Bool:
    if len(bitmap) == 0:
        return True
    var byte = i // 8
    if byte >= len(bitmap):
        return False
    return ((bitmap[byte] >> UInt8(i % 8)) & 1) == 1


def bit_fill_valid(mut bitmap: List[UInt8], count: Int):
    """Materialise a bitmap in which the first `count` entries are all valid."""
    var whole = count // 8
    for _ in range(whole):
        bitmap.append(0xFF)
    if count % 8:
        bitmap.append((UInt8(1) << UInt8(count % 8)) - 1)


def bit_set(mut bitmap: List[UInt8], i: Int, value: Bool):
    var byte = i // 8
    while len(bitmap) <= byte:
        bitmap.append(0)
    if value:
        bitmap[byte] |= UInt8(1) << UInt8(i % 8)
    else:
        bitmap[byte] &= ~(UInt8(1) << UInt8(i % 8))


# ── foreign buffers ────────────────────────────────────────────────────────


comptime _ReleaseFn = def(Int) thin abi("C") -> None
"""An `ArrowArray`'s `release`, called with the struct's address."""


struct ForeignBuffers(Movable):
    """Producer memory that imported `ArrayData` buffers point into.

    `arrow_mlake.carrow_import` creates one per imported root `ArrowArray`
    and every node whose buffers borrow from that array holds an
    `ArcPointer` to it, so the producer's `release` runs exactly once, when
    the last such node is destroyed. `storage` is the struct's own
    allocation when this library made it (a stream's `get_next` target) and
    is freed after the release.
    """

    var array: Int
    """Address of the root `ArrowArray` this owns."""
    var free_storage: Bool

    def __init__(out self, array: Int, free_storage: Bool):
        self.array = array
        self.free_storage = free_storage

    def __init__(out self, *, deinit move: Self):
        self.array = move.array
        self.free_storage = move.free_storage

    def __deinit__(deinit self):
        if self.array == 0:
            return
        var rel = Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=self.array + 64
        )
        if rel[] != 0:
            var slot = Pointer[_ReleaseFn, MutUntrackedOrigin](
                unsafe_from_address=self.array + 64
            )
            slot[](self.array)
        if self.free_storage:
            Pointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=self.array
            ).unsafe_free()


def _list_layout[T: Copyable]() -> Tuple[Int, Int]:
    """Byte offsets of a `List[T]`'s data pointer and length.

    `List` keeps these private and has moved them between releases: 1.2 has
    `_data`, `_len`, `_capacity`; later nightlies hold the pointer and the
    capacity in an `_storage: Allocation`. Both keep the length in `_len` and
    free nothing at capacity 0. Read by reflection so that either compiles;
    `(-1, -1)` for a layout this does not know, and `borrowed_list` copies.
    """
    comptime L = List[T]
    comptime names = reflect[L].field_names()
    comptime for i in range(reflect[L].field_count()):
        comptime if names[i] == "_data":
            return (
                reflect[L].field_offset[name="_data"](),
                reflect[L].field_offset[name="_len"](),
            )
        elif names[i] == "_storage":
            comptime S = reflect[L].field["_storage"].T
            comptime A = reflect[S].field["_alloc"].T
            return (
                reflect[L].field_offset[name="_storage"]()
                + reflect[S].field_offset[name="_alloc"]()
                + reflect[A].field_offset[name="_ptr"](),
                reflect[L].field_offset[name="_len"](),
            )
    return (-1, -1)


def borrowed_list[T: Copyable](addr: Int, count: Int) -> List[T]:
    """A `List` whose elements are `count` values at `addr`, not owned.

    The list has capacity 0, which is what makes it safe to drop: `List`
    frees its storage only when it has capacity, and the first push, resize
    or reserve moves the elements into a fresh allocation of its own (the
    foreign memory is left as it was). A copy is an ordinary owned list.
    Writing an element in place writes the producer's memory, so imported
    arrays are read-only. On a `List` layout it does not recognise, this
    copies instead.
    """
    var out = List[T]()
    if count <= 0:
        return out^
    comptime at = _list_layout[T]()
    comptime if at[0] < 0:
        out.reserve(count)
        var src = Pointer[T, ImmUntrackedOrigin](unsafe_from_address=addr)
        for i in range(count):
            out.append(src[unsafe_offset=i].copy())
        return out^
    else:
        var base = Int(Pointer(to=out))
        Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=base + at[0]
        )[] = addr
        Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=base + at[1]
        )[] = count
        if (
            Int(out.unsafe_ptr()) != addr
            or len(out) != count
            or out.capacity() != 0
        ):
            abort("arrow_mlake: borrowed_list does not match List's layout")
        return out^


def is_borrowed(list: List[_]) -> Bool:
    """True for a `borrowed_list`: elements present, no storage owned."""
    return len(list) > 0 and list.capacity() == 0


# ── array data ─────────────────────────────────────────────────────────────


struct ArrayData(Copyable, Movable):
    """One Arrow array. Children are indices into the owning `ArrayArena`."""

    var type: ArrowType
    var name: String
    var nullable: Bool
    var field_id: Int32
    var length: Int
    var null_count: Int
    var validity: List[UInt8]
    var offsets: List[Int32]
    var large_offsets: List[Int64]
    var values: List[UInt8]
    var children: List[Int]
    var foreign: Optional[ArcPointer[ForeignBuffers]]
    """Set on an array imported without copying: the producer memory its
    buffers point into, released when the last array holding it dies.

    The buffers are valid only while this array (or another holding the same
    `foreign`) is alive. Moving one out (`a.values^`, `swap`) and keeping it
    past the array is a use-after-free; `a.copy()` or `own_buffers()` first.
    """

    def __init__(out self, var type: ArrowType, var name: String):
        self.type = type^
        self.name = name^
        self.nullable = True
        self.field_id = -1
        self.length = 0
        self.null_count = 0
        self.validity = List[UInt8]()
        self.offsets = List[Int32]()
        self.large_offsets = List[Int64]()
        self.values = List[UInt8]()
        self.children = List[Int]()
        self.foreign = None

    def __init__(out self):
        self = ArrayData(ArrowType(AT_NULL), String())

    def __init__(out self, *, copy: Self):
        self.type = copy.type.copy()
        self.name = copy.name.copy()
        self.nullable = copy.nullable
        self.field_id = copy.field_id
        self.length = copy.length
        self.null_count = copy.null_count
        self.validity = copy.validity.copy()
        self.offsets = copy.offsets.copy()
        self.large_offsets = copy.large_offsets.copy()
        self.values = copy.values.copy()
        self.children = copy.children.copy()
        # Copying a borrowed list allocates, so a copy owns everything.
        self.foreign = None

    def __init__(out self, *, deinit move: Self):
        self.type = move.type^
        self.name = move.name^
        self.nullable = move.nullable
        self.field_id = move.field_id
        self.length = move.length
        self.null_count = move.null_count
        self.validity = move.validity^
        self.offsets = move.offsets^
        self.large_offsets = move.large_offsets^
        self.values = move.values^
        self.children = move.children^
        self.foreign = move.foreign^

    def own_buffers(mut self):
        """Copy any borrowed buffer into memory of this array's own."""
        if not self.foreign:
            return
        self.validity = self.validity.copy()
        self.offsets = self.offsets.copy()
        self.large_offsets = self.large_offsets.copy()
        self.values = self.values.copy()
        self.foreign = None

    def is_valid(self, i: Int) -> Bool:
        return bit_get(Span(self.validity), i)


struct ArrayArena(Copyable, Defaultable, Movable):
    """Owns every `ArrayData` of a batch; children are referenced by index."""

    var nodes: List[ArrayData]

    def __init__(out self):
        self.nodes = List[ArrayData]()

    def __init__(out self, *, copy: Self):
        self.nodes = copy.nodes.copy()

    def __init__(out self, *, deinit move: Self):
        self.nodes = move.nodes^

    def add(mut self, var node: ArrayData) -> Int:
        self.nodes.append(node^)
        return len(self.nodes) - 1

    def graft(mut self, mut other: Self, root: Int) -> Int:
        """Move every node of `other` in here, keeping its order; return where
        `root` ended up.

        This is what lets an array be *built* somewhere else and still land in a
        caller-chosen place in this arena. Children are indices, so a subtree
        built on its own — starting at index 0 — needs every one of them shifted
        by however many nodes are already here. Because the nodes keep their
        relative order, the result is byte for byte the arena that building the
        same subtree straight into this one would have produced.

        That equivalence is the whole point: it is what makes it safe to build
        several top-level fields at once and concatenate them in field order
        afterwards, whatever order they actually finished in.

        `other` is left holding that many empty arrays — an `ArrayData` owns its
        buffers, so this moves them out one slot at a time rather than copying.
        """
        var base = len(self.nodes)
        for k in range(len(other.nodes)):
            var node = ArrayData()
            swap(node, other.nodes[k])
            for c in range(len(node.children)):
                node.children[c] += base
            self.nodes.append(node^)
        return base + root

    def __len__(self) -> Int:
        return len(self.nodes)


# ── typed views over a values buffer ───────────────────────────────────────


# Arrow buffers are little-endian and densely packed, which is exactly the
# layout the target already uses, so element `i` is one unaligned load. These
# used to assemble each value a byte at a time; on a column scan that cost
# eight bounds-checked loads and eight shifts per value, and it showed —
# comparing a million `int64`s took 8 ns each instead of 1.
#
# `alignment=1` is not optional: a values buffer starts wherever its `List`
# was allocated and a page boundary need not be 8-byte aligned.


@always_inline
def load_i32(buf: Span[UInt8, _], i: Int) -> Int32:
    return buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_load[alignment=1](i)


@always_inline
def load_i64(buf: Span[UInt8, _], i: Int) -> Int64:
    return buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_load[alignment=1](i)


@always_inline
def load_u64(buf: Span[UInt8, _], i: Int) -> UInt64:
    return buf.unsafe_ptr().unsafe_bitcast[UInt64]().unsafe_load[alignment=1](i)


@always_inline
def load_f32(buf: Span[UInt8, _], i: Int) -> Float32:
    return (
        buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_load[alignment=1](i)
    )


@always_inline
def load_f64(buf: Span[UInt8, _], i: Int) -> Float64:
    return (
        buf.unsafe_ptr().unsafe_bitcast[Float64]().unsafe_load[alignment=1](i)
    )


def store_u32(mut buf: List[UInt8], v: UInt32):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))


def store_u64(mut buf: List[UInt8], v: UInt64):
    for k in range(8):
        buf.append(UInt8((v >> UInt64(8 * k)) & 0xFF))
