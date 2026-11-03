"""Arrow IPC streams: `pixi run test` runs these after the main suite.

The corpus is `sample_arena`, the same sixteen columns the C Data Interface
tests use, so every buffer layout goes through the writer and back through
the reader and is compared with `assert_same_array`. Whether *pyarrow* agrees
is `pixi run verify-ipc`.
"""

from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from build_arrays import SAMPLE_ROWS, int64_array, sample_arena, utf8_array
from carrow_check import assert_preorder, assert_same_array
from arrow_mlake import (
    AT_INT64,
    LZ4_FRAME,
    ZSTD,
    ArrayArena,
    IpcCodecs,
    IpcWriter,
    RecordBatch,
    read_ipc_file,
    read_ipc_stream,
    write_ipc_stream,
)


def sample_batch() raises -> RecordBatch:
    var built = sample_arena()
    var b = RecordBatch()
    b.arena = built[0].copy()
    b.roots = built[1].copy()
    b.num_rows = SAMPLE_ROWS
    return b^


def _assert_same_batch(a: RecordBatch, b: RecordBatch, label: String) raises:
    assert_equal(a.num_rows, b.num_rows)
    assert_equal(a.num_columns(), b.num_columns())
    for c in range(a.num_columns()):
        var at = String(label, ".", a.name(c))
        assert_same_array(a.arena, a.roots[c], b.arena, b.roots[c], at)
        assert_preorder(b.arena, b.roots[c], at)


def test_every_layout_round_trips_through_ipc() raises:
    var b = sample_batch()
    var bytes = write_ipc_stream([b.copy()])
    var s = read_ipc_stream(Span(bytes))
    assert_equal(len(s.batches), 1)
    _assert_same_batch(b, s.batches[0], "ipc")


def test_a_stream_holds_several_batches_and_a_schema() raises:
    var b = sample_batch()
    var w = IpcWriter(b)
    for _ in range(3):
        w.write(b)
    var bytes = w^.finish()
    var s = read_ipc_stream(Span(bytes))
    assert_equal(len(s.batches), 3)
    for i in range(3):
        _assert_same_batch(b, s.batches[i], String("batch ", i))
    assert_equal(s.schema.num_columns(), 16)
    assert_equal(s.schema.num_rows, 0)
    for c in range(16):
        assert_equal(String(s.schema.type(c)), String(b.type(c)))
        assert_equal(s.schema.name(c), b.name(c))
        assert_equal(s.schema.column(c).length, 0)


def test_a_schema_alone_is_a_stream() raises:
    var w = IpcWriter(sample_batch())
    var bytes = w^.finish()
    var s = read_ipc_stream(Span(bytes))
    assert_equal(len(s.batches), 0)
    assert_equal(s.schema.num_columns(), 16)


def test_messages_and_bodies_are_8_byte_aligned() raises:
    var bytes = write_ipc_stream([sample_batch()])
    # Walk the framing: every message starts on a multiple of 8.
    var pos = 0
    var messages = 0
    while True:
        assert_equal(pos % 8, 0)
        assert_equal(Int(bytes[pos]), 0xFF)
        var size = 0
        for i in range(4):
            size |= Int(bytes[pos + 4 + i]) << (8 * i)
        if size == 0:
            break
        assert_equal(size % 8, 0)
        messages += 1
        pos += 8 + size
        if messages == 2:
            # The batch's body runs to the end-of-stream marker.
            assert_equal((len(bytes) - 8 - pos) % 8, 0)
            pos = len(bytes) - 8
    assert_equal(messages, 2)


def test_field_ids_and_extensions_travel() raises:
    var b = sample_batch()
    b.arena.nodes[b.roots[0]].field_id = 7
    var bytes = write_ipc_stream([b.copy()])
    var s = read_ipc_stream(Span(bytes))
    assert_equal(Int(s.batches[0].column(0).field_id), 7)
    assert_equal(Int(s.batches[0].column(1).field_id), -1)
    assert_equal(s.batches[0].column(8).type.extension, "arrow.uuid")
    assert_equal(s.schema.column(8).type.extension, "arrow.uuid")


def test_an_empty_batch_round_trips() raises:
    var b = RecordBatch()
    b.roots.append(b.arena.add(int64_array(String("x"), 0, 0, 0)))
    b.roots.append(b.arena.add(utf8_array(String("s"), 0, False)))
    var bytes = write_ipc_stream([b.copy()])
    var s = read_ipc_stream(Span(bytes))
    _assert_same_batch(b, s.batches[0], "empty")


def test_a_batch_that_does_not_match_the_schema_is_refused() raises:
    var w = IpcWriter(sample_batch())
    var other = RecordBatch()
    other.num_rows = 2
    other.roots.append(other.arena.add(int64_array(String("x"), 2, 0, 0)))
    with assert_raises(contains="columns"):
        w.write(other)


def test_a_truncated_stream_is_rejected() raises:
    var bytes = write_ipc_stream([sample_batch()])
    for cut in [5, 20, len(bytes) // 2, len(bytes) - 9]:
        var short = List[UInt8](Span(bytes)[:cut])
        with assert_raises():
            _ = read_ipc_stream(Span(short))


def test_garbage_metadata_is_rejected_not_followed() raises:
    var junk: List[UInt8] = [0xFF, 0xFF, 0xFF, 0xFF, 16, 0, 0, 0]
    for _ in range(16):
        junk.append(0xFF)
    with assert_raises(contains="flatbuf"):
        _ = read_ipc_stream(Span(junk))


struct ToyRle(IpcCodecs):
    """Run-length pairs (count, byte): zeros shrink, varied bytes grow, so a
    batch has buffers stored both compressed and as they are. Only
    `LZ4_FRAME` is accepted, to see the codec number arrive."""

    @staticmethod
    def compress(codec: Int, data: Span[UInt8, _]) raises -> List[UInt8]:
        if codec != LZ4_FRAME:
            raise Error("toy: no codec ", codec)
        var out = List[UInt8]()
        var i = 0
        while i < len(data):
            var j = i
            while j < len(data) and j - i < 255 and data[j] == data[i]:
                j += 1
            out.append(UInt8(j - i))
            out.append(data[i])
            i = j
        return out^

    @staticmethod
    def decompress(
        codec: Int, data: Span[UInt8, _], size: Int
    ) raises -> List[UInt8]:
        if codec != LZ4_FRAME:
            raise Error("toy: no codec ", codec)
        var out = List[UInt8]()
        for k in range(0, len(data), 2):
            for _ in range(Int(data[k])):
                out.append(data[k + 1])
        return out^


def _compressed_stream(b: RecordBatch, batches: Int) raises -> List[UInt8]:
    var w = IpcWriter(b)
    for _ in range(batches):
        w.write[ToyRle](b, codec=LZ4_FRAME)
    return w^.finish()


def test_compressed_batches_round_trip() raises:
    var b = sample_batch()
    var bytes = _compressed_stream(b, 2)
    var s = read_ipc_stream[ToyRle](Span(bytes))
    assert_equal(len(s.batches), 2)
    for i in range(2):
        _assert_same_batch(b, s.batches[i], String("compressed ", i))
    # Compression shrank the stream: the sample's bitmaps and offsets are
    # mostly zeros.
    assert_true(len(bytes) < len(write_ipc_stream([b.copy(), b.copy()])))


def test_a_compressed_stream_needs_a_codec() raises:
    var bytes = _compressed_stream(sample_batch(), 1)
    with assert_raises(contains="compressed (LZ4_FRAME)"):
        _ = read_ipc_stream(Span(bytes))


def test_writing_a_codec_the_codecs_lack_is_refused() raises:
    var b = sample_batch()
    var w = IpcWriter(b)
    with assert_raises(contains="toy: no codec 1"):
        w.write[ToyRle](b, codec=ZSTD)


def _as_file(stream: List[UInt8]) -> List[UInt8]:
    """An IPC file around `stream`: magic, padding, the stream, a stand-in
    footer, its length, the magic again."""
    var out = List[UInt8]()
    for c in "ARROW1".as_bytes():
        out.append(c)
    out.append(0)
    out.append(0)
    for x in stream:
        out.append(x)
    var footer: List[UInt8] = [1, 2, 3, 4, 5, 6, 7, 8]
    for x in footer:
        out.append(x)
    out.append(8)
    out.append(0)
    out.append(0)
    out.append(0)
    for c in "ARROW1".as_bytes():
        out.append(c)
    return out^


def test_an_ipc_file_is_read_through_its_stream() raises:
    var b = sample_batch()
    var plain = _as_file(write_ipc_stream([b.copy()]))
    var f = read_ipc_file(Span(plain))
    assert_equal(len(f.batches), 1)
    _assert_same_batch(b, f.batches[0], "file")
    var packed = _as_file(_compressed_stream(b, 3))
    var g = read_ipc_file[ToyRle](Span(packed))
    assert_equal(len(g.batches), 3)
    _assert_same_batch(b, g.batches[2], "compressed file")


def test_a_stream_is_not_a_file() raises:
    var bytes = write_ipc_stream([sample_batch()])
    with assert_raises(contains="ARROW1"):
        _ = read_ipc_file(Span(bytes))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
