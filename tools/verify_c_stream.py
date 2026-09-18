"""Consume a Mojo-produced ArrowArrayStream with stock pyarrow.

The producer side of the C Data Interface, checked against an implementation
nobody here wrote. Run via the `verify-c-stream` task, which builds the
library first.
"""

import ctypes
import gc
import sys

import pyarrow as pa

EXPECTED = [[1, 2, 3], [4, 5], [6]]


def main(lib_path: str) -> int:
    lib = ctypes.CDLL(lib_path)
    lib.am_stream_of_three_batches.restype = ctypes.c_void_p
    lib.am_stream_of_three_batches.argtypes = []

    addr = lib.am_stream_of_three_batches()
    if not addr:
        print("FAIL: producer returned NULL", file=sys.stderr)
        return 1

    reader = pa.RecordBatchReader._import_from_c(addr)
    assert reader.schema == pa.schema([pa.field("n", pa.int64(), nullable=False)]), (
        f"schema mismatch: {reader.schema}"
    )

    got = [b.column(0).to_pylist() for b in reader]
    assert got == EXPECTED, f"batches: {got!r} != {EXPECTED!r}"

    # Batch boundaries are the point: a reader that concatenated would pass a
    # row-count check and lose what the producer was saying about splits.
    assert [len(b) for b in got] == [3, 2, 1]

    del reader
    gc.collect()

    # Draining leaves nothing for the stream's release to free, so this path
    # exercises the empty case. The abandoned one below exercises the other.
    addr2 = lib.am_stream_of_three_batches()
    reader2 = pa.RecordBatchReader._import_from_c(addr2)
    first = reader2.read_next_batch()
    assert first.column(0).to_pylist() == [1, 2, 3]
    del reader2
    gc.collect()
    # Two batches went unread; their release ran inside the stream's release.
    # A double free here would abort the process, which is the assertion.

    print(f"ok: {len(got)} batches, {sum(len(b) for b in got)} rows, "
          f"then a stream abandoned after one batch")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
