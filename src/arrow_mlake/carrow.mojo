"""The Arrow C Data Interface — this library's public output contract.

`export_c` turns one `ArrayData` tree into the two C structs every Arrow
implementation understands:

```c
struct ArrowSchema { const char* format; const char* name; const char* metadata;
                     int64_t flags; int64_t n_children; ArrowSchema** children;
                     ArrowSchema* dictionary; void (*release)(ArrowSchema*);
                     void* private_data; };
struct ArrowArray  { int64_t length; int64_t null_count; int64_t offset;
                     int64_t n_buffers; int64_t n_children; const void** buffers;
                     ArrowArray** children; ArrowArray* dictionary;
                     void (*release)(ArrowArray*); void* private_data; };
```

Each export is **two** allocations — one for the whole schema tree, one for the
whole array tree — laid out as: the structs, then the pointer arrays, then the
format and name strings and the metadata blocks, then copies of every buffer.
The root's `private_data` is the base of its block and its `release` frees the
lot; children carry a release that only clears itself, because a consumer
releases the root and the root owns everything. Both callbacks are `abi("C")`,
so they can be called from C, Python or Rust.

Buffers are copied rather than borrowed, which is what makes the export
outlive the `RecordBatch` it came from and makes `ExportedArray` safe to hand
to a runtime that will release it whenever it likes.

```mojo
var e = export_c(batch.arena, batch.roots[0])
print(e.array, e.schema)   # two addresses, ready for _import_from_c
e.into_raw()               # hand ownership to the consumer
```

`arrow_mlake.carrow_import` is the other direction — a foreign `ArrowSchema`
and `ArrowArray` read back into an `ArrayData`, and `ArrowArrayStream`
iterated — and it implements the consumer's half of the same release
convention.
"""

from std.memory import unsafe_memcpy, unsafe_memset
from memory_region import BumpAllocator, HeapRegion

from arrow_mlake.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_LARGE_BINARY,
    AT_LARGE_LIST,
    AT_LARGE_UTF8,
    AT_LIST,
    AT_MAP,
    AT_NULL,
    AT_STRUCT,
    AT_UTF8,
    ArrayArena,
    ArrayData,
)

comptime ARROW_FLAG_NULLABLE: Int64 = 2


@fieldwise_init
struct CArrowSchema(Copyable, Movable):
    """The C `ArrowSchema`, field for field. Pointers are held as addresses."""

    var format: Int
    var name: Int
    var metadata: Int
    var flags: Int64
    var n_children: Int64
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int


@fieldwise_init
struct CArrowArray(Copyable, Movable):
    """The C `ArrowArray`, field for field. Pointers are held as addresses."""

    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    var buffers: Int
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int


def release_array(p: Pointer[CArrowArray, MutUntrackedOrigin]) abi("C") -> None:
    """Free the whole exported array tree. `private_data` is the block."""
    var block = p[].private_data
    p[].release = 0
    if block != 0:
        var base = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=block
        )
        base.unsafe_free()


def release_schema(
    p: Pointer[CArrowSchema, MutUntrackedOrigin]
) abi("C") -> None:
    """Free the whole exported schema tree."""
    var block = p[].private_data
    p[].release = 0
    if block != 0:
        var base = Pointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=block
        )
        base.unsafe_free()


def release_child_array(
    p: Pointer[CArrowArray, MutUntrackedOrigin]
) abi("C") -> None:
    """A child is owned by its root; releasing one only marks it released."""
    p[].release = 0


def release_child_schema(
    p: Pointer[CArrowSchema, MutUntrackedOrigin]
) abi("C") -> None:
    p[].release = 0


def n_buffers_for_type(i: Int) -> Int:
    """How many buffers an `ArrowArray` of this type id must carry.

    The export writes this many and `arrow_mlake.carrow_import` rejects a
    producer that sends any other number, so the count is stated once rather
    than twice.
    """
    if i == AT_NULL:
        return 0
    if i == AT_STRUCT:
        return 1
    if i == AT_LIST or i == AT_LARGE_LIST or i == AT_MAP:
        return 2
    if (
        i == AT_UTF8
        or i == AT_BINARY
        or i == AT_LARGE_UTF8
        or i == AT_LARGE_BINARY
    ):
        return 3
    return 2


def n_buffers_for(a: ArrayData) -> Int:
    return n_buffers_for_type(a.type.id)


def _align8(n: Int) -> Int:
    return (n + 7) & ~7


struct _Block(Movable):
    """One export's bytes, carved front to back.

    A thin shim over `memory_region.BumpAllocator` so the export keeps its own
    vocabulary. What the library adds is the distinction the C Data Interface
    forces on us: `take` hands back an **offset**, and `address_of` turns one
    into a pointer for the moment a value has to be written into a
    `CArrowArray` — which is the only place an address belongs, because it is
    the only place the consumer is guaranteed to be this process.

    Anywhere that distinction does not matter, an offset is the better thing
    to hold: it is what a region mapped somewhere else can still resolve.
    """

    var bump: BumpAllocator[HeapRegion]

    def __init__(out self, size: Int):
        self.bump = BumpAllocator[HeapRegion](HeapRegion(size))
        # Zeroed for the same reason it always was: the tail of a buffer whose
        # source list is short is read by the consumer, and padding between
        # buffers should not be whatever the allocator left there.
        unsafe_memset(
            ptr=self.bytes_at(0), value=0, count=self.bump.region.size()
        )

    def __init__(out self, *, deinit move: Self):
        self.bump = move.bump^

    def address(self) -> Int:
        """Where the block starts in this process."""
        return self.bump.region.base()

    def take(mut self, n: Int) raises -> Int:
        """Claim `n` bytes; returns their offset from the block's start."""
        return self.bump.claim(n)

    def address_of(self, offset: Int) -> Int:
        """`offset` as a pointer value, for storing in a C structure."""
        return self.bump.unsafe_address(offset)

    def into_raw(deinit self) -> Int:
        """The bytes are the consumer's now — its release callback frees them.

        Spelled rather than implied: the export used to end by dropping the
        block, which read like a free and was the opposite of one.
        """
        return self.bump^.into_raw()

    def close(deinit self):
        """Give the bytes back, for an export that did not finish."""
        self.bump^.close()

    def bytes_at(self, offset: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self.bump.unsafe_ptr(offset)

    def words_at(self, offset: Int) -> Pointer[Int64, MutUntrackedOrigin]:
        # The C Data Interface lays its structs out as Int64 fields, which is
        # this file's business rather than the allocator's.
        return self.bump.unsafe_ptr(offset).unsafe_bitcast[Int64]()

    def put_bytes(mut self, data: Span[UInt8, _]) raises -> Int:
        return self.bump.append(data)

    def put_cstring(mut self, text: StringSlice) raises -> Int:
        var b = text.as_bytes()
        var at = self.take(len(b) + 1)
        if len(b):
            unsafe_memcpy(
                dest=self.bytes_at(at), src=b.unsafe_ptr(), count=len(b)
            )
        self.bytes_at(at)[unsafe_offset=len(b)] = 0
        return at


def _metadata_bytes(a: ArrayData) -> List[UInt8]:
    """Arrow's C metadata encoding: `n`, then (`len`, key, `len`, value)*."""
    var out = List[UInt8]()
    if not a.type.extension:
        return out^
    var keys: List[String] = [String("ARROW:extension:name")]
    var vals: List[String] = [a.type.extension.copy()]
    _put_i32(out, Int32(len(keys)))
    for i in range(len(keys)):
        _put_i32(out, Int32(keys[i].byte_length()))
        out.extend(keys[i].as_bytes())
        _put_i32(out, Int32(vals[i].byte_length()))
        out.extend(vals[i].as_bytes())
    return out^


def _put_i32(mut out: List[UInt8], v: Int32):
    var u = UInt32(v)
    for k in range(4):
        out.append(UInt8((u >> UInt32(8 * k)) & 0xFF))


def _validity_bytes(a: ArrayData) -> Int:
    if a.null_count == 0:
        return 0
    return (a.length + 7) // 8


def _values_bytes(a: ArrayData) -> Int:
    var i = a.type.id
    if i == AT_BOOL:
        return (a.length + 7) // 8
    if (
        i == AT_UTF8
        or i == AT_BINARY
        or i == AT_LARGE_UTF8
        or i == AT_LARGE_BINARY
    ):
        return len(a.values)
    if (
        i == AT_STRUCT
        or i == AT_LIST
        or i == AT_LARGE_LIST
        or i == AT_MAP
        or i == AT_NULL
    ):
        return 0
    return a.type.fixed_width() * a.length


def _offsets_bytes(a: ArrayData) -> Int:
    var i = a.type.id
    if i == AT_UTF8 or i == AT_BINARY or i == AT_LIST or i == AT_MAP:
        return 4 * (a.length + 1)
    if i == AT_LARGE_UTF8 or i == AT_LARGE_BINARY or i == AT_LARGE_LIST:
        return 8 * (a.length + 1)
    return 0


def _collect(arena: ArrayArena, root: Int, mut order: List[Int]):
    """Depth-first order of an array and its children, parents first."""
    var stack: List[Int] = [root]
    while len(stack):
        var node = stack.pop()
        order.append(node)
        ref kids = arena.nodes[node].children
        for k in range(len(kids)):
            stack.append(kids[len(kids) - 1 - k])


struct ExportedArray(Movable):
    """An exported pair of C structs, still owned by Mojo until `into_raw`."""

    var array: Int
    """Address of the `ArrowArray`."""
    var schema: Int
    """Address of the `ArrowSchema`."""
    var _owned: Bool

    def __init__(out self, array: Int, schema: Int):
        self.array = array
        self.schema = schema
        self._owned = True

    def __init__(out self, *, deinit move: Self):
        self.array = move.array
        self.schema = move.schema
        self._owned = move._owned

    def into_raw(mut self) -> Tuple[Int, Int]:
        """Give the two structs to the consumer; they must release them."""
        self._owned = False
        return (self.array, self.schema)

    def release(mut self):
        if not self._owned:
            return
        self._owned = False
        var a = Pointer[CArrowArray, MutUntrackedOrigin](
            unsafe_from_address=self.array
        )
        if a[].release != 0:
            release_array(a)
        var s = Pointer[CArrowSchema, MutUntrackedOrigin](
            unsafe_from_address=self.schema
        )
        if s[].release != 0:
            release_schema(s)

    def __deinit__(deinit self):
        self.release()


def _export_schema(
    arena: ArrayArena, root: Int, order: List[Int]
) raises -> Int:
    var n = len(order)
    var index = Dict[Int, Int]()
    for i in range(n):
        index[order[i]] = i
    # Size the block.
    var size = n * 72  # nine 8-byte words per ArrowSchema
    for node in order:
        ref a = arena.nodes[node]
        size += _align8(len(a.children) * 8)
        size += _align8(a.type.format().byte_length() + 1)
        size += _align8(a.name.byte_length() + 1)
        var md = _metadata_bytes(a)
        if len(md):
            size += _align8(len(md))
    var blk = _Block(size + 64)
    var structs = blk.take(n * 72)
    for i in range(n):
        ref a = arena.nodes[order[i]]
        var kids = len(a.children)
        var kidptr = 0
        if kids:
            kidptr = blk.take(kids * 8)
        var fmt = blk.put_cstring(a.type.format())
        var nm = blk.put_cstring(a.name)
        var md = _metadata_bytes(a)
        var mdp = 0
        if len(md):
            mdp = blk.put_bytes(Span(md))
        var w = blk.words_at(structs + i * 72)
        # Pointers from here on: a consumer of the C Data Interface reads
        # these in this process, so offsets have to become addresses.
        w[unsafe_offset=0] = Int64(blk.address_of(fmt))
        w[unsafe_offset=1] = Int64(blk.address_of(nm))
        w[unsafe_offset=2] = Int64(blk.address_of(mdp)) if mdp else 0
        w[unsafe_offset=3] = ARROW_FLAG_NULLABLE if a.nullable else 0
        w[unsafe_offset=4] = Int64(kids)
        w[unsafe_offset=5] = Int64(blk.address_of(kidptr)) if kids else 0
        w[unsafe_offset=6] = 0
        w[unsafe_offset=7] = 0
        w[unsafe_offset=8] = Int64(blk.address()) if i == 0 else 0
        _store_schema_release(blk.address_of(structs + i * 72), i == 0)
        if kids:
            var kp = blk.words_at(kidptr)
            for k in range(kids):
                kp[unsafe_offset=k] = Int64(
                    blk.address_of(structs + index[a.children[k]] * 72)
                )
    var addr = blk.address_of(structs)
    # The consumer owns the bytes from here; its release callback frees them.
    _ = blk^.into_raw()
    return addr


comptime SchemaReleaseFn = def(
    Pointer[CArrowSchema, MutUntrackedOrigin]
) thin abi("C") -> None
comptime ArrayReleaseFn = def(
    Pointer[CArrowArray, MutUntrackedOrigin]
) thin abi("C") -> None


def _store_schema_release(struct_addr: Int, is_root: Bool):
    """Write the `release` function pointer into a C `ArrowSchema` (word 7)."""
    var slot = Pointer[SchemaReleaseFn, MutUntrackedOrigin](
        unsafe_from_address=struct_addr + 56
    )
    slot[] = release_schema if is_root else release_child_schema


def _store_array_release(struct_addr: Int, is_root: Bool):
    """Write the `release` function pointer into a C `ArrowArray` (word 8)."""
    var slot = Pointer[ArrayReleaseFn, MutUntrackedOrigin](
        unsafe_from_address=struct_addr + 64
    )
    slot[] = release_array if is_root else release_child_array


def _export_array(arena: ArrayArena, root: Int, order: List[Int]) raises -> Int:
    var n = len(order)
    var index = Dict[Int, Int]()
    for i in range(n):
        index[order[i]] = i
    var size = n * 80  # ten 8-byte words per ArrowArray
    for node in order:
        ref a = arena.nodes[node]
        size += _align8(len(a.children) * 8)
        size += _align8(n_buffers_for(a) * 8)
        size += _align8(_validity_bytes(a)) + 8
        size += _align8(_offsets_bytes(a)) + 8
        size += _align8(_values_bytes(a)) + 8
    var blk = _Block(size + 64)
    var structs = blk.take(n * 80)
    for i in range(n):
        ref a = arena.nodes[order[i]]
        var kids = len(a.children)
        var nbuf = n_buffers_for(a)
        var kidptr = 0
        if kids:
            kidptr = blk.take(kids * 8)
        var bufptr = 0
        if nbuf:
            bufptr = blk.take(nbuf * 8)
        var bp = blk.words_at(bufptr) if nbuf else blk.words_at(structs)
        if nbuf > 0:
            # buffer 0 is always validity, NULL when there are no nulls
            if a.null_count > 0:
                var vb = _validity_bytes(a)
                var at = blk.take(vb)
                var n = vb if vb < len(a.validity) else len(a.validity)
                if n:
                    unsafe_memcpy(
                        dest=blk.bytes_at(at),
                        src=a.validity.unsafe_ptr(),
                        count=n,
                    )
                bp[unsafe_offset=0] = Int64(blk.address_of(at))
            else:
                bp[unsafe_offset=0] = 0
        var slot = 1
        var ob = _offsets_bytes(a)
        if ob > 0:
            var at = blk.take(ob)
            var wide = ob == 8 * (a.length + 1)
            # Both lists are already native-endian, and the C Data Interface
            # hands over native-endian buffers, so the little-endian byte
            # assembly this replaces was a memcpy written out one shift at a
            # time. An offsets buffer is `length + 1` wide and a short source
            # list leaves zeros behind it, which is what the block gives.
            var n: Int
            if wide:
                n = 8 * len(a.large_offsets)
                if n > ob:
                    n = ob
                if n:
                    unsafe_memcpy(
                        dest=blk.bytes_at(at),
                        src=a.large_offsets.unsafe_ptr().unsafe_bitcast[
                            UInt8
                        ](),
                        count=n,
                    )
            else:
                n = 4 * len(a.offsets)
                if n > ob:
                    n = ob
                if n:
                    unsafe_memcpy(
                        dest=blk.bytes_at(at),
                        src=a.offsets.unsafe_ptr().unsafe_bitcast[UInt8](),
                        count=n,
                    )
            bp[unsafe_offset=slot] = Int64(blk.address_of(at))
            slot += 1
        var vb2 = _values_bytes(a)
        if slot < nbuf:
            if vb2 > 0:
                var at = blk.take(vb2)
                var n = vb2 if vb2 < len(a.values) else len(a.values)
                if n:
                    unsafe_memcpy(
                        dest=blk.bytes_at(at),
                        src=a.values.unsafe_ptr(),
                        count=n,
                    )
                bp[unsafe_offset=slot] = Int64(blk.address_of(at))
            else:
                bp[unsafe_offset=slot] = Int64(blk.address_of(blk.take(1)))
            slot += 1
        var w = blk.words_at(structs + i * 80)
        w[unsafe_offset=0] = Int64(a.length)
        w[unsafe_offset=1] = Int64(a.null_count)
        w[unsafe_offset=2] = 0
        w[unsafe_offset=3] = Int64(nbuf)
        w[unsafe_offset=4] = Int64(kids)
        w[unsafe_offset=5] = Int64(blk.address_of(bufptr)) if nbuf else 0
        w[unsafe_offset=6] = Int64(blk.address_of(kidptr)) if kids else 0
        w[unsafe_offset=7] = 0
        w[unsafe_offset=8] = 0
        w[unsafe_offset=9] = Int64(blk.address()) if i == 0 else 0
        _store_array_release(blk.address_of(structs + i * 80), i == 0)
        if kids:
            var kp = blk.words_at(kidptr)
            for k in range(kids):
                kp[unsafe_offset=k] = Int64(
                    blk.address_of(structs + index[a.children[k]] * 80)
                )
    var addr = blk.address_of(structs)
    # The consumer owns the bytes from here; its release callback frees them.
    _ = blk^.into_raw()
    return addr


def export_c(arena: ArrayArena, root: Int) raises -> ExportedArray:
    """Export one array of an arena over the Arrow C Data Interface."""
    var order = List[Int]()
    _collect(arena, root, order)
    var schema = _export_schema(arena, root, order)
    var array = _export_array(arena, root, order)
    return ExportedArray(array, schema)
