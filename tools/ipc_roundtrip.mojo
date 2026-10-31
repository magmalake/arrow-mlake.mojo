"""The Mojo side of `pixi run verify-ipc`; `tools/verify_ipc.py` drives it.

    ipc-roundtrip write OUT        the test corpus as an IPC stream
    ipc-roundtrip copy IN OUT      read a stream, write it back out
    ipc-roundtrip check IN         read a stream, compare it with the corpus
"""

from std.sys import argv

from build_arrays import SAMPLE_ROWS, sample_arena
from carrow_check import assert_same_array
from arrow_mlake import RecordBatch, read_ipc_stream, write_ipc_stream


def _sample() raises -> RecordBatch:
    var built = sample_arena()
    var b = RecordBatch()
    b.arena = built[0].copy()
    b.roots = built[1].copy()
    b.num_rows = SAMPLE_ROWS
    return b^


def _read(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def _write(path: String, data: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: ipc-roundtrip write OUT | copy IN OUT | check IN")
    var mode = String(args[1])
    if mode == "write":
        _write(String(args[2]), write_ipc_stream([_sample()]))
    elif mode == "copy":
        var s = read_ipc_stream(Span(_read(String(args[2]))))
        if len(s.batches) == 0:
            raise Error("copy: the stream has no batches")
        _write(String(args[3]), write_ipc_stream(s.batches))
        print(
            "copied",
            len(s.batches),
            "batches of",
            s.schema.num_columns(),
            "columns",
        )
    elif mode == "check":
        var s = read_ipc_stream(Span(_read(String(args[2]))))
        var want = _sample()
        if len(s.batches) != 1:
            raise Error("check: expected one batch, got ", len(s.batches))
        ref got = s.batches[0]
        if got.num_columns() != want.num_columns():
            raise Error("check: column count differs")
        for c in range(want.num_columns()):
            assert_same_array(
                want.arena,
                want.roots[c],
                got.arena,
                got.roots[c],
                String("pyarrow.", want.name(c)),
            )
        print("checked", want.num_columns(), "columns")
    else:
        raise Error("unknown mode ", mode)
