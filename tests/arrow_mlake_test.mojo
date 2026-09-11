"""The arrow-mlake.mojo test suite. `pixi run test`.

Everything here builds its own Arrow arrays (`tests/build_arrays.mojo`) rather
than reading a file, so the coverage is a property of this library and not of
somebody else's decoder. The cross-language gate — pyarrow producing arrays
that this imports, and comparing what it gets back — is `pixi run
verify-c-import`, which needs Python and lives outside the unit tests.
"""

from build_arrays import (
    SAMPLE_ROWS,
    add_list,
    add_map,
    add_struct,
    bool_array,
    int64_array,
    null_array,
    put_i32,
    sample_arena,
    utf8_array,
    uuid_array,
)
from carrow_check import (
    StreamSource,
    assert_preorder,
    assert_same_array,
    buffer_address,
    child_array,
    exported_pair,
    make_struct_arena,
    permuted_arena,
    preorder_size,
    set_word,
    word,
)
from arrow_mlake import (
    AT_BINARY,
    AT_BOOL,
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
    AT_UINT8,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UTF8,
    AT_DATE32,
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    TU_SECOND,
    ArrayArena,
    ArrayData,
    ArrowType,
    ImportedArray,
    ImportedStream,
    export_stream,
    RecordBatch,
    array_i64,
    array_str,
    at_decimal,
    at_fixed,
    at_time,
    at_timestamp,
    bit_fill_valid,
    bit_get,
    bit_set,
    bitmap_bytes,
    export_c,
    extension_name,
    import_batch_c,
    import_c,
    load_f32,
    load_f64,
    load_i32,
    load_i64,
    load_u64,
    n_buffers_for_type,
    parquet_field_id,
    parse_format,
    store_u32,
    store_u64,
    unit_name,
)
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)


def _cstring(addr: Int) -> String:
    var p = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    return String(unsafe_from_utf8_ptr=p)


def _schema_format(schema: Int) -> String:
    return _cstring(Int(word(schema, 0)))


# ── arrow_mlake.arrow ──────────────────────────────────────────────────────


def test_every_type_has_a_format_string() raises:
    """The format strings, spelled out, against the C Data Interface's own
    table — so a typo is a failing test rather than a silently misread buffer
    on the other side of the interface."""
    assert_equal(ArrowType(AT_NULL).format(), "n")
    assert_equal(ArrowType(AT_BOOL).format(), "b")
    assert_equal(ArrowType(AT_INT8).format(), "c")
    assert_equal(ArrowType(AT_UINT8).format(), "C")
    assert_equal(ArrowType(AT_INT16).format(), "s")
    assert_equal(ArrowType(AT_UINT16).format(), "S")
    assert_equal(ArrowType(AT_INT32).format(), "i")
    assert_equal(ArrowType(AT_UINT32).format(), "I")
    assert_equal(ArrowType(AT_INT64).format(), "l")
    assert_equal(ArrowType(AT_UINT64).format(), "L")
    assert_equal(ArrowType(AT_FLOAT16).format(), "e")
    assert_equal(ArrowType(AT_FLOAT32).format(), "f")
    assert_equal(ArrowType(AT_FLOAT64).format(), "g")
    assert_equal(ArrowType(AT_UTF8).format(), "u")
    assert_equal(ArrowType(AT_LARGE_UTF8).format(), "U")
    assert_equal(ArrowType(AT_BINARY).format(), "z")
    assert_equal(ArrowType(AT_LARGE_BINARY).format(), "Z")
    assert_equal(at_fixed(16).format(), "w:16")
    assert_equal(at_decimal(38, 9).format(), "d:38,9")
    assert_equal(ArrowType(AT_DATE32).format(), "tdD")
    assert_equal(at_time(TU_MILLI).format(), "ttm")
    assert_equal(at_time(TU_NANO).format(), "ttn")
    assert_equal(at_timestamp(TU_MICRO, String("UTC")).format(), "tsu:UTC")
    assert_equal(at_timestamp(TU_NANO, String()).format(), "tsn:")
    assert_equal(ArrowType(AT_LIST).format(), "+l")
    assert_equal(ArrowType(AT_LARGE_LIST).format(), "+L")
    assert_equal(ArrowType(AT_STRUCT).format(), "+s")
    assert_equal(ArrowType(AT_MAP).format(), "+m")


def test_types_write_themselves_out() raises:
    assert_equal(String(ArrowType(AT_INT32)), "int32")
    assert_equal(String(ArrowType(AT_FLOAT64)), "double")
    assert_equal(String(at_decimal(10, 2)), "decimal128(10, 2)")
    assert_equal(String(at_fixed(16)), "fixed_size_binary[16]")
    assert_equal(String(at_time(TU_SECOND)), "time32[s]")
    assert_equal(String(at_time(TU_MICRO)), "time64[us]")
    assert_equal(
        String(at_timestamp(TU_MICRO, String("UTC"))), "timestamp[us, tz=UTC]"
    )
    assert_equal(String(at_timestamp(TU_NANO, String())), "timestamp[ns]")
    var uuid = at_fixed(16)
    uuid.extension = String("arrow.uuid")
    assert_equal(String(uuid), "fixed_size_binary[16] <arrow.uuid>")
    assert_equal(unit_name(TU_MILLI), "ms")


def test_fixed_widths_and_nesting() raises:
    assert_equal(ArrowType(AT_INT8).fixed_width(), 1)
    assert_equal(ArrowType(AT_FLOAT16).fixed_width(), 2)
    assert_equal(ArrowType(AT_DATE32).fixed_width(), 4)
    assert_equal(ArrowType(AT_INT64).fixed_width(), 8)
    assert_equal(at_decimal(38, 9).fixed_width(), 16)
    assert_equal(at_fixed(7).fixed_width(), 7)
    assert_equal(ArrowType(AT_UTF8).fixed_width(), 0)
    assert_true(ArrowType(AT_LIST).is_nested())
    assert_true(ArrowType(AT_MAP).is_nested())
    assert_true(ArrowType(AT_STRUCT).is_nested())
    assert_false(ArrowType(AT_BINARY).is_nested())


def test_validity_bitmaps() raises:
    assert_equal(bitmap_bytes(0), 0)
    assert_equal(bitmap_bytes(1), 1)
    assert_equal(bitmap_bytes(8), 1)
    assert_equal(bitmap_bytes(9), 2)
    var bm = List[UInt8]()
    bit_set(bm, 11, True)
    assert_equal(len(bm), 2)
    assert_true(bit_get(Span(bm), 11))
    assert_false(bit_get(Span(bm), 10))
    bit_set(bm, 11, False)
    assert_false(bit_get(Span(bm), 11))
    # An empty bitmap means "no nulls", not "all null".
    var none = List[UInt8]()
    assert_true(bit_get(Span(none), 0))
    assert_true(bit_get(Span(none), 1000))
    # Past the end of a non-empty bitmap is out of range, and invalid.
    assert_false(bit_get(Span(bm), 64))
    var filled = List[UInt8]()
    bit_fill_valid(filled, 11)
    assert_equal(len(filled), 2)
    for i in range(11):
        assert_true(bit_get(Span(filled), i), String("bit ", i))
    assert_false(bit_get(Span(filled), 11))


def test_typed_loads_read_unaligned_buffers() raises:
    """The values buffer starts wherever its `List` was allocated, so every
    load is unaligned by construction; the odd leading byte here is what makes
    that true in the test as well."""
    var buf = List[UInt8]()
    buf.append(0xEE)  # one byte of padding, so nothing below is 4- or 8-aligned
    store_u32(buf, UInt32(0xDEADBEEF))
    store_u32(buf, UInt32(1))
    var body = Span(buf)[1:]
    assert_equal(load_i32(body, 0), Int32(-559038737))
    assert_equal(load_i32(body, 1), Int32(1))

    var wide = List[UInt8]()
    wide.append(0xEE)
    store_u64(wide, UInt64(0x0123456789ABCDEF))
    store_u64(wide, UInt64(Int64(-2)))
    var wb = Span(wide)[1:]
    assert_equal(load_u64(wb, 0), UInt64(0x0123456789ABCDEF))
    assert_equal(load_i64(wb, 1), Int64(-2))

    var f = List[UInt8]()
    f.append(0xEE)
    store_u32(f, UInt32(0x3F800000))  # 1.0f
    store_u32(f, UInt32(0x40000000))  # 2.0f
    var fb = Span(f)[1:]
    assert_equal(load_f32(fb, 0), Float32(1.0))
    assert_equal(load_f32(fb, 1), Float32(2.0))
    var d = List[UInt8]()
    d.append(0xEE)
    store_u64(d, UInt64(0x4000000000000000))  # 2.0
    assert_equal(load_f64(Span(d)[1:], 0), Float64(2.0))


def test_grafting_an_arena_shifts_its_children() raises:
    """`graft` is what lets a subtree be built on its own and land anywhere.

    The promise is that the result is the arena building it in place would
    have produced, so the check is exactly that: two arenas, one grafted and
    one built directly, compared node for node."""
    var target = ArrayArena()
    _ = target.add(int64_array(String("filler"), 3, 0, 0))
    _ = target.add(int64_array(String("filler2"), 3, 0, 0))
    var side = ArrayArena()
    var side_root = add_struct(side, String("st"), 5)
    var n_side = len(side.nodes)
    var landed = target.graft(side, side_root)
    assert_equal(landed, 2 + side_root)
    assert_equal(len(target.nodes), 2 + n_side)
    assert_equal(len(side.nodes), n_side)  # emptied in place, not shortened
    var direct = ArrayArena()
    _ = direct.add(int64_array(String("filler"), 3, 0, 0))
    _ = direct.add(int64_array(String("filler2"), 3, 0, 0))
    var direct_root = add_struct(direct, String("st"), 5)
    assert_equal(landed, direct_root)
    assert_same_array(target, landed, direct, direct_root, "grafted")
    for k in range(len(target.nodes[landed].children)):
        assert_equal(
            target.nodes[landed].children[k],
            direct.nodes[direct_root].children[k],
        )


def test_an_array_copy_owns_its_own_buffers() raises:
    var a = int64_array(String("i"), 4, 10, 0)
    var b = a.copy()
    b.values[0] = 0xFF
    assert_equal(a.values[0], 10)
    assert_equal(b.name, "i")
    assert_true(a.is_valid(0))


# ── arrow_mlake.carrow — the export ────────────────────────────────────────


def test_buffer_counts_follow_the_type() raises:
    assert_equal(n_buffers_for_type(AT_NULL), 0)
    assert_equal(n_buffers_for_type(AT_STRUCT), 1)
    assert_equal(n_buffers_for_type(AT_LIST), 2)
    assert_equal(n_buffers_for_type(AT_LARGE_LIST), 2)
    assert_equal(n_buffers_for_type(AT_MAP), 2)
    assert_equal(n_buffers_for_type(AT_UTF8), 3)
    assert_equal(n_buffers_for_type(AT_LARGE_BINARY), 3)
    assert_equal(n_buffers_for_type(AT_INT64), 2)
    assert_equal(n_buffers_for_type(AT_BOOL), 2)


def test_export_lays_out_a_primitive_column() raises:
    var arena = ArrayArena()
    var root = arena.add(int64_array(String("i64"), 6, -5, 3))
    var e = export_c(arena, root)
    assert_equal(_schema_format(e.schema), "l")
    assert_equal(_cstring(Int(word(e.schema, 1))), "i64")
    assert_equal(word(e.schema, 4), 0)  # no children
    assert_equal(word(e.array, 0), 6)  # length
    assert_equal(word(e.array, 1), 2)  # every third of six is null
    assert_equal(word(e.array, 3), 2)  # validity + values
    assert_true(word(e.array, 8) != 0)  # a release callback
    var bufs = Int(word(e.array, 5))
    assert_true(word(bufs, 0) != 0, "nulls, so a validity buffer")
    var values = Pointer[Int64, ImmUntrackedOrigin](
        unsafe_from_address=Int(word(bufs, 1))
    )
    assert_equal(values[unsafe_offset=0], -5)
    assert_equal(values[unsafe_offset=1], -4)
    e.release()


def test_export_omits_a_validity_buffer_when_nothing_is_null() raises:
    """Arrow says buffer 0 is NULL when `null_count` is zero, and a consumer
    is entitled to skip reading it — writing an all-ones bitmap there instead
    would be legal but is not what this does."""
    var arena = ArrayArena()
    var root = arena.add(int64_array(String("plain"), 6, 1, 0))
    var e = export_c(arena, root)
    assert_equal(word(e.array, 1), 0)
    assert_equal(word(Int(word(e.array, 5)), 0), 0)
    e.release()


def test_export_lays_out_a_map_column() raises:
    """A map is `+m` over a two-field `+s`, and the key is not nullable."""
    var arena = ArrayArena()
    var root = add_map(arena, String("m"), SAMPLE_ROWS)
    var e = export_c(arena, root)
    assert_equal(_schema_format(e.schema), "+m")
    assert_equal(_cstring(Int(word(e.schema, 1))), "m")
    assert_equal(word(e.schema, 4), 1)
    var kv = Int(word(Int(word(e.schema, 5)), 0))
    assert_equal(_schema_format(kv), "+s")
    assert_equal(word(kv, 4), 2)
    var key = Int(word(Int(word(kv, 5)), 0))
    assert_equal(_schema_format(key), "u")
    assert_equal(_cstring(Int(word(key, 1))), "key")
    assert_equal(word(key, 3), 0)  # flags: not nullable
    assert_equal(word(e.array, 0), SAMPLE_ROWS)
    assert_equal(word(e.array, 3), 2)  # validity + offsets
    assert_equal(word(e.array, 4), 1)
    e.release()


def test_export_writes_extension_metadata() raises:
    """`arrow.uuid` travels as an `ARROW:extension:name` metadata pair, which
    is the one part of a type the format string cannot carry."""
    var arena = ArrayArena()
    var root = arena.add(uuid_array(String("uuid"), 4))
    var e = export_c(arena, root)
    assert_equal(_schema_format(e.schema), "w:16")
    var md = Int(word(e.schema, 2))
    assert_true(md != 0, "a uuid column should carry extension metadata")
    assert_equal(extension_name(md), "arrow.uuid")
    e.release()


def test_export_release_is_idempotent() raises:
    var built = sample_arena()
    for c in range(len(built[1])):
        var e = export_c(built[0], built[1][c])
        e.release()
        e.release()


# ── the round trip ─────────────────────────────────────────────────────────


def _round_trip(arena: ArrayArena, root: Int, label: StringSlice) raises:
    """Export one array, import it back, and compare both halves.

    Values *and* layout: `assert_same_array` walks the two trees together, and
    `assert_preorder` looks at the arena's own numbering, which the walk by
    construction cannot.
    """
    var pair = exported_pair(arena, root)
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    var got = import_c(back, pair[0], pair[1])
    owned.release()
    assert_same_array(arena, root, back, got, label)
    assert_preorder(back, got, label)
    assert_equal(len(back.nodes), preorder_size(arena, root))


def test_every_layout_round_trips() raises:
    """Every column of the corpus, out through the exporter and back.

    That is nulls and no nulls, every primitive width, utf8, large utf8 and
    binary, decimal, a temporal type, an extension type, the null type with no
    buffers at all, and all four nested shapes.
    """
    var built = sample_arena()
    for c in range(len(built[1])):
        var root = built[1][c]
        _round_trip(
            built[0], root, String("sample.", built[0].nodes[root].name)
        )
    assert_equal(len(built[1]), 16)


def test_an_import_lays_the_arena_out_in_pre_order() raises:
    """A map column is four nodes; imported, they are 0, 1, 2, 3 in DFS order.
    """
    var arena = ArrayArena()
    var root = add_map(arena, String("m"), SAMPLE_ROWS)
    var pair = exported_pair(arena, root)
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    var got = import_c(back, pair[0], pair[1])
    owned.release()
    assert_equal(got, 0)
    assert_equal(len(back.nodes), 4)
    assert_equal(back.nodes[0].type.format(), "+m")
    assert_equal(back.nodes[1].type.format(), "+s")
    assert_equal(back.nodes[1].children[0], 2)
    assert_equal(back.nodes[1].children[1], 3)
    assert_preorder(back, got, "m")
    # And a second import into the same arena appends after the first.
    var pair2 = exported_pair(arena, root)
    var owned2 = ImportedArray(pair2[0], pair2[1])
    var got2 = import_c(back, pair2[0], pair2[1])
    owned2.release()
    assert_equal(got2, 4)
    assert_preorder(back, got2, "m again")


def test_the_round_trip_check_catches_a_corrupted_buffer() raises:
    """The negative control for the values half.

    Three separate corruptions of an *exported* array — one values byte, one
    offset, one validity bit — each of which the importer must carry faithfully
    into its `ArrayData` and the comparator must then report. A round trip that
    passed these would be proving nothing.
    """
    var arena = ArrayArena()
    var ints = arena.add(int64_array(String("i64"), 9, -5, 3))
    var strs = arena.add(utf8_array(String("s"), 9, False))

    # One byte of the int64 values buffer.
    var pair = exported_pair(arena, ints)
    var values = buffer_address(pair[0], 1)
    var vp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=values)
    vp[unsafe_offset=9] ^= 0x40
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    var got = import_c(back, pair[0], pair[1])
    owned.release()
    with assert_raises(contains="values byte"):
        assert_same_array(arena, ints, back, got, "i64")

    # One validity bit of the same column.
    var pair2 = exported_pair(arena, ints)
    var validity = buffer_address(pair2[0], 0)
    var bp = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=validity)
    bp[unsafe_offset=0] ^= 0x01
    var owned2 = ImportedArray(pair2[0], pair2[1])
    var back2 = ArrayArena()
    var got2 = import_c(back2, pair2[0], pair2[1])
    owned2.release()
    with assert_raises(contains="null_count"):
        assert_same_array(arena, ints, back2, got2, "i64")

    # One offset of the utf8 column, which moves a string boundary.
    var pair3 = exported_pair(arena, strs)
    var offs = buffer_address(pair3[0], 1)
    var op = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=offs)
    op[unsafe_offset=2] += 1
    var owned3 = ImportedArray(pair3[0], pair3[1])
    var back3 = ArrayArena()
    var got3 = import_c(back3, pair3[0], pair3[1])
    owned3.release()
    with assert_raises(contains="offset"):
        assert_same_array(arena, strs, back3, got3, "s")


def test_the_layout_check_catches_a_permuted_arena() raises:
    """The negative control for the layout half.

    Renumber an imported arena back to front without touching a value: the
    lockstep walk still passes, because it starts at the root and follows the
    indices wherever they lead, and `assert_preorder` fails, because it is the
    only half that looks at the numbers.
    """
    var arena = ArrayArena()
    var root = add_struct(arena, String("st"), SAMPLE_ROWS)
    var pair = exported_pair(arena, root)
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    var got = import_c(back, pair[0], pair[1])
    owned.release()
    assert_true(len(back.nodes) > 1, "a flat array cannot show a permutation")
    var moved = permuted_arena(back, got)
    assert_same_array(back, got, moved[0], moved[1], "permuted")
    with assert_raises(contains="in pre-order"):
        assert_preorder(moved[0], moved[1], "permuted")


# ── arrow_mlake.carrow_import ──────────────────────────────────────────────


def test_import_honours_a_producers_offset() raises:
    """A producer's `offset` is a slice, and `ArrayData` has nowhere to put it.

    pyarrow exports a sliced array as the whole buffer plus an offset, so an
    importer that ignored the field would read the wrong elements — and one
    that materialised the window but forgot the child would read the wrong
    strings. Both are checked here by slicing an export by hand and comparing
    against the elements the original holds at those positions.
    """
    var rows = 12
    var arena = ArrayArena()
    var ints = arena.add(int64_array(String("i64"), rows, -5, 3))
    var strs = arena.add(utf8_array(String("s"), rows, False))
    var want_i64 = array_i64(arena.nodes[ints])
    var want_str = array_str(arena.nodes[strs])
    for skip in [1, 3, 7]:
        var pair = exported_pair(arena, ints)
        set_word(pair[0], 2, Int64(skip))  # offset
        set_word(pair[0], 0, Int64(rows - skip))  # length
        set_word(pair[0], 1, -1)  # null_count: "not computed"
        var owned = ImportedArray(pair[0], pair[1])
        var back = ArrayArena()
        var got = import_c(back, pair[0], pair[1])
        owned.release()
        assert_equal(back.nodes[got].length, rows - skip)
        var have = array_i64(back.nodes[got])
        for i in range(rows - skip):
            assert_equal(have[1][i], want_i64[1][i + skip], String("valid ", i))
            if have[1][i]:
                assert_equal(have[0][i], want_i64[0][i + skip])

        var pair2 = exported_pair(arena, strs)
        set_word(pair2[0], 2, Int64(skip))
        set_word(pair2[0], 0, Int64(rows - skip))
        var owned2 = ImportedArray(pair2[0], pair2[1])
        var back2 = ArrayArena()
        var got2 = import_c(back2, pair2[0], pair2[1])
        owned2.release()
        assert_equal(back2.nodes[got2].offsets[0], 0)
        var have2 = array_str(back2.nodes[got2])
        for i in range(rows - skip):
            assert_equal(have2[1][i], want_str[1][i + skip])
            if have2[1][i]:
                assert_equal(have2[0][i], want_str[0][i + skip])


def test_import_honours_an_offset_into_a_list() raises:
    """The child of a sliced list has to be sliced by the parent's offsets,
    not by the parent's own window — an importer that passed the window
    straight down would read the right number of values from the wrong place.
    """
    var arena = ArrayArena()
    var root = add_list(arena, String("li"), SAMPLE_ROWS, False)
    var whole = ArrayArena()
    var pair0 = exported_pair(arena, root)
    var owned0 = ImportedArray(pair0[0], pair0[1])
    var full = import_c(whole, pair0[0], pair0[1])
    owned0.release()
    var skip = 4
    var pair = exported_pair(arena, root)
    set_word(pair[0], 2, Int64(skip))
    set_word(pair[0], 0, Int64(SAMPLE_ROWS - skip))
    set_word(pair[0], 1, -1)
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    var got = import_c(back, pair[0], pair[1])
    owned.release()
    assert_equal(back.nodes[got].length, SAMPLE_ROWS - skip)
    assert_equal(back.nodes[got].offsets[0], 0)
    # Cell `i` of the slice holds cell `i + skip` of the whole.
    var sliced = array_i64(back.nodes[back.nodes[got].children[0]])
    var entire = array_i64(whole.nodes[whole.nodes[full].children[0]])
    var base = Int(whole.nodes[full].offsets[skip])
    for i in range(len(sliced[0])):
        assert_equal(sliced[0][i], entire[0][base + i], String("value ", i))


def test_import_rejects_a_wrong_buffer_count() raises:
    """A malformed `n_buffers` raises, and says which format it disagreed with.
    """
    var arena = ArrayArena()
    var root = arena.add(utf8_array(String("s"), 5, False))
    var pair = exported_pair(arena, root)
    set_word(pair[0], 3, 2)  # utf8 needs three
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    with assert_raises(contains="u needs 3 buffers"):
        _ = import_c(back, pair[0], pair[1])
    owned.release()


def test_import_rejects_a_wrong_child_count() raises:
    var arena = ArrayArena()
    var root = add_list(arena, String("li"), SAMPLE_ROWS, False)
    var pair = exported_pair(arena, root)
    set_word(pair[0], 4, 0)  # a list needs exactly one child
    set_word(pair[1], 4, 0)
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    with assert_raises(contains="needs 1 children"):
        _ = import_c(back, pair[0], pair[1])
    owned.release()


def test_import_rejects_a_short_child() raises:
    """A list whose offsets run past its child is a truncated buffer, and the
    one truncation the interface makes detectable: it carries no buffer sizes,
    so a short `utf8` data buffer cannot be seen, but a short child array can.
    """
    var arena = ArrayArena()
    var li = add_list(arena, String("li"), SAMPLE_ROWS, False)
    var pair = exported_pair(arena, li)
    var child = child_array(pair[0], 0)
    set_word(child, 0, 1)  # the child now claims a single element
    var owned = ImportedArray(pair[0], pair[1])
    var back = ArrayArena()
    with assert_raises(contains="run past the end of its child"):
        _ = import_c(back, pair[0], pair[1])
    owned.release()

    # And a struct whose child is shorter than the struct itself.
    var st = add_struct(arena, String("st"), SAMPLE_ROWS)
    var pair2 = exported_pair(arena, st)
    set_word(child_array(pair2[0], 0), 0, 2)
    var owned2 = ImportedArray(pair2[0], pair2[1])
    var back2 = ArrayArena()
    with assert_raises(contains="shorter than the struct"):
        _ = import_c(back2, pair2[0], pair2[1])
    owned2.release()


def test_import_rejects_a_released_pair() raises:
    """Importing after release is a use-after-free; the null callback catches it.
    """
    var arena = ArrayArena()
    var root = arena.add(int64_array(String("i64"), 5, 0, 0))
    var pair = exported_pair(arena, root)
    var owned = ImportedArray(pair[0], pair[1])
    owned.release()
    owned.release()  # idempotent
    var back = ArrayArena()
    with assert_raises(contains="released ArrowArray"):
        _ = import_c(back, pair[0], pair[1])


def test_import_parses_the_formats_we_write() raises:
    """Every format string `ArrowType.format` emits parses back to itself."""
    var types: List[ArrowType] = [
        ArrowType(AT_NULL),
        ArrowType(AT_BOOL),
        ArrowType(AT_INT8),
        ArrowType(AT_UINT8),
        ArrowType(AT_INT16),
        ArrowType(AT_UINT16),
        ArrowType(AT_INT32),
        ArrowType(AT_UINT32),
        ArrowType(AT_INT64),
        ArrowType(AT_UINT64),
        ArrowType(AT_FLOAT16),
        ArrowType(AT_FLOAT32),
        ArrowType(AT_FLOAT64),
        ArrowType(AT_UTF8),
        ArrowType(AT_LARGE_UTF8),
        ArrowType(AT_BINARY),
        ArrowType(AT_LARGE_BINARY),
        at_fixed(16),
        at_decimal(38, 9),
        ArrowType(AT_DATE32),
        at_time(TU_SECOND),
        at_time(TU_MILLI),
        at_time(TU_MICRO),
        at_time(TU_NANO),
        at_timestamp(TU_MICRO, String("UTC")),
        at_timestamp(TU_NANO, String()),
        ArrowType(AT_LIST),
        ArrowType(AT_LARGE_LIST),
        ArrowType(AT_STRUCT),
        ArrowType(AT_MAP),
    ]
    for want in types:
        var got = parse_format(want.format())
        assert_equal(got.format(), want.format())
        assert_equal(String(got), String(want))
    # A negative scale, which Arrow allows and our writer never emits.
    assert_equal(parse_format("d:10,-2").scale, -2)
    assert_equal(parse_format("d:10,2,128").precision, 10)


def test_import_names_the_formats_it_will_not_read() raises:
    """An unknown format is an error carrying the string, never a guess."""
    var bad: List[String] = [
        String("+w:3"),  # fixed size list
        String("+ud:0,1"),  # dense union
        String("+r"),  # run-end encoded
        String("vu"),  # string view
        String("tdm"),  # date64
        String("tDs"),  # duration
        String("tiM"),  # interval
        String("d:10,2,256"),  # decimal256
        String("q"),  # nothing at all
        String(""),
    ]
    for f in bad:
        with assert_raises(contains="arrow_mlake.carrow"):
            _ = parse_format(f)
        if f:
            with assert_raises(contains=String(f)):
                _ = parse_format(f)


def test_metadata_keys_are_read_and_bounded() raises:
    """The C metadata block is an unbounded `const char*`, so the counts in it
    are sanity-checked before they are used as loop bounds — a garbage `n`
    would otherwise be an unbounded read of somebody else's memory."""
    var md = List[UInt8]()
    put_i32(md, 2)
    _put_kv(md, String("ARROW:extension:name"), String("arrow.uuid"))
    _put_kv(md, String("PARQUET:field_id"), String("-17"))
    var at = Int(md.unsafe_ptr())
    assert_equal(extension_name(at), "arrow.uuid")
    assert_equal(parquet_field_id(at), Int32(-17))
    assert_equal(extension_name(0), "")
    assert_equal(parquet_field_id(0), Int32(-1))
    var hostile = List[UInt8]()
    put_i32(hostile, 1 << 30)
    with assert_raises(contains="implausible metadata key count"):
        _ = extension_name(Int(hostile.unsafe_ptr()))
    assert_equal(parquet_field_id(Int(hostile.unsafe_ptr())), Int32(-1))
    _ = md^
    _ = hostile^


def _put_kv(mut md: List[UInt8], key: String, value: String):
    put_i32(md, Int32(key.byte_length()))
    md.extend(key.as_bytes())
    put_i32(md, Int32(value.byte_length()))
    md.extend(value.as_bytes())


# ── RecordBatch ────────────────────────────────────────────────────────────


def test_import_batch_unwraps_a_struct() raises:
    """A struct array becomes a `RecordBatch`, one column per field."""
    var built = make_struct_arena(7, 500)
    var pair = exported_pair(built[0], built[1])
    var owned = ImportedArray(pair[0], pair[1])
    var batch = owned.into_batch()
    owned.release()
    assert_equal(batch.num_columns(), 1)
    assert_equal(batch.num_rows, 7)
    assert_equal(batch.name(0), "n")
    var got = batch.column_i64(0)
    assert_equal(got[0][0], 500)
    assert_equal(got[0][1], 501)
    assert_false(got[1][2])  # every third is null


def test_import_batch_keeps_a_bare_column_whole() raises:
    """Anything that is not a `+s` root is a one-column batch, which is what a
    caller importing a bare column wants."""
    var arena = ArrayArena()
    var root = arena.add(int64_array(String("i64"), 6, 3, 0))
    var pair = exported_pair(arena, root)
    var owned = ImportedArray(pair[0], pair[1])
    var batch = import_batch_c(pair[0], pair[1])
    owned.release()
    assert_equal(batch.num_columns(), 1)
    assert_equal(batch.num_rows, 6)
    assert_equal(batch.name(0), "i64")


def test_import_batch_refuses_a_struct_with_its_own_nulls() raises:
    """A `RecordBatch` is a list of columns and not an array, so there is
    nowhere to put a struct's row-level nulls; dropping them silently would be
    a wrong answer."""
    var arena = ArrayArena()
    var root = add_struct(arena, String("st"), 8)
    var validity = List[UInt8]()
    for i in range(8):
        bit_set(validity, i, i != 0)
    arena.nodes[root].validity = validity^
    arena.nodes[root].null_count = 1
    var pair = exported_pair(arena, root)
    var owned = ImportedArray(pair[0], pair[1])
    with assert_raises(contains="cannot be a RecordBatch"):
        _ = import_batch_c(pair[0], pair[1])
    owned.release()


def test_record_batch_accessors() raises:
    """The typed accessors, over a batch assembled from the corpus."""
    var built = sample_arena()
    var batch = RecordBatch()
    batch.num_rows = SAMPLE_ROWS
    batch.roots = built[1].copy()
    batch.arena = built[0].copy()
    assert_equal(batch.num_columns(), 16)
    assert_equal(batch.name(0), "i64")
    assert_equal(String(batch.type(2)), "double")
    var ints = batch.column_i64(0)
    assert_equal(ints[0][0], -5)
    assert_false(ints[1][2])  # every third is null
    assert_equal(batch.column_f64(2)[0][3], 3.5)
    assert_true(batch.column_bool(3)[0][0])
    assert_false(batch.column_bool(3)[0][1])
    assert_equal(batch.column_str(4)[0][0], "")
    assert_equal(batch.column_str(4)[0][1], "value-1")
    # A date32 widens to Int64 like any other integer column.
    assert_equal(batch.column_i64(10)[0][0], 17486)
    # And the batch exports one column at a time.
    var e = batch.export_c(0)
    assert_equal(_schema_format(e.schema), "l")
    assert_equal(word(e.array, 0), SAMPLE_ROWS)
    e.release()
    assert_equal(batch.child(batch.roots[14], 0).name, "a")


def test_record_batch_accessors_name_the_type_they_refused() raises:
    var arena = ArrayArena()
    var batch = RecordBatch()
    batch.num_rows = 4
    batch.roots = [
        arena.add(utf8_array(String("s"), 4, False)),
        arena.add(bool_array(String("b"), 4)),
        arena.add(null_array(String("nul"), 4)),
    ]
    batch.arena = arena^
    with assert_raises(contains="string is not an integer"):
        _ = batch.column_i64(0)
    with assert_raises(contains="bool is not floating point"):
        _ = batch.column_f64(1)
    with assert_raises(contains="string is not boolean"):
        _ = batch.column_bool(0)
    with assert_raises(contains="null is not a byte array"):
        _ = batch.column_str(2)


# ── ArrowArrayStream ───────────────────────────────────────────────────────


def test_arrow_array_stream_iterates_to_the_end() raises:
    """Three batches out of a real C producer, in order, then a clean end."""
    var src = StreamSource(3, 4, -1)
    var stream = ImportedStream(src.address())
    assert_equal(stream.format(), "+s")
    assert_true(stream.schema_address() != 0)
    var seen = 0
    var rows = 0
    while True:
        var got = stream.next()
        if not got:
            break
        ref batch = got.value()
        assert_equal(batch.num_columns(), 1)
        assert_equal(batch.num_rows, 4 + seen)
        var values = batch.column_i64(0)
        assert_equal(values[0][0], Int64(100 * seen))
        rows += batch.num_rows
        seen += 1
    assert_equal(seen, 3)
    assert_equal(rows, 4 + 5 + 6)
    # Past the end it stays finished rather than asking again.
    assert_false(Bool(stream.next()))
    stream.release()
    stream.release()
    src.free_stream_storage()


def test_arrow_array_stream_reports_a_producer_error() raises:
    """`get_next` returning non-zero is an error, not the end of the stream.

    Conflating the two would turn a failed scan into a short one, which is the
    kind of wrong answer that never gets noticed, so the message
    `get_last_error` returns is carried into the raise.
    """
    var src = StreamSource(3, 4, 1)
    var stream = ImportedStream(src.address())
    var first = stream.next()
    assert_true(Bool(first))
    with assert_raises(contains="the producer stopped early"):
        _ = stream.next()
    stream.release()
    src.free_stream_storage()


def test_arrow_array_stream_releases_what_it_never_read() raises:
    """A stream dropped half way frees the batches it never handed over."""
    var src = StreamSource(4, 2, -1)
    var stream = ImportedStream(src.address())
    var first = stream.next()
    assert_true(Bool(first))
    stream.release()
    src.free_stream_storage()


def _stream_batch(mut arena: ArrayArena, var values: List[Int64]) raises -> Int:
    """One batch for `export_stream`: `struct<n: int64>`, non-null."""
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


def _three_batch_stream(mut arena: ArrayArena) raises -> Int:
    """1..3, 4..5, 6 — uneven, so a consumer that assumed a fixed batch size
    would be caught."""
    var roots = List[Int]()
    roots.append(_stream_batch(arena, [Int64(1), Int64(2), Int64(3)]))
    roots.append(_stream_batch(arena, [Int64(4), Int64(5)]))
    roots.append(_stream_batch(arena, [Int64(6)]))
    return export_stream(arena, roots)


def test_export_stream_round_trips_through_the_importer() raises:
    """Our producer, read by our consumer, batch boundaries intact.

    `pixi run verify-c-stream` is the gate that matters — it puts pyarrow on
    the consuming end — but this runs with no Python and catches a break in
    the same commit that causes it.
    """
    var arena = ArrayArena()
    var stream = ImportedStream(_three_batch_stream(arena))

    var sizes = List[Int]()
    var total = Int64(0)
    while True:
        var got = stream.next()
        if not got:
            break
        ref batch = got.value()
        assert_equal(batch.num_columns(), 1)
        sizes.append(batch.num_rows)
        var values = batch.column_i64(0)
        for i in range(len(values[0])):
            total += values[0][i]

    assert_equal(len(sizes), 3)
    assert_equal(sizes[0], 3)
    assert_equal(sizes[1], 2)
    assert_equal(sizes[2], 1)
    assert_equal(total, Int64(21))
    stream.release()


def test_export_stream_ends_with_a_released_array() raises:
    """The end of a stream is a NULL `release`, not an error code.

    A consumer that treated the end as an error would report a failed scan;
    one that treated an error as the end would report a short one. Both are
    silent, so the distinction is asserted rather than assumed.
    """
    var arena = ArrayArena()
    var stream = ImportedStream(_three_batch_stream(arena))
    for _ in range(3):
        assert_true(Bool(stream.next()))
    assert_false(Bool(stream.next()))
    assert_false(Bool(stream.next()))
    stream.release()


def test_export_stream_frees_the_batches_nobody_took() raises:
    """Abandoning a stream after one batch must not double free the rest.

    `release` frees what the cursor never reached; a consumer already holds
    what it did reach. Getting this wrong aborts the process, so the
    assertion is that the test finishes at all.
    """
    var arena = ArrayArena()
    var stream = ImportedStream(_three_batch_stream(arena))
    assert_true(Bool(stream.next()))
    stream.release()
    stream.release()


def test_export_stream_refuses_an_empty_stream() raises:
    """No arrays means no schema, and a schemaless stream is not a stream."""
    var arena = ArrayArena()
    with assert_raises(contains="at least one array"):
        _ = export_stream(arena, List[Int]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
