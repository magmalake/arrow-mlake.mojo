#!/usr/bin/env python3
"""pyarrow and arrow-mlake read each other's Arrow IPC streams.

Both directions, because each catches what the other can't:

* **Mojo writes, pyarrow reads.** `ipc-roundtrip write` puts the sixteen-column
  test corpus in a stream. pyarrow reads it and validates every buffer
  (`validate(full=True)`), checks a few values, and writes it back out with
  its own writer. `ipc-roundtrip check` then compares pyarrow's bytes with the
  corpus, column by column.
* **pyarrow writes, Mojo reads.** pyarrow builds batches of every type
  arrow-mlake holds, sliced arrays and nesting included, and writes a stream.
  `ipc-roundtrip copy` reads it and writes it again, and pyarrow checks that
  what comes back equals what it sent.

    python tools/verify_ipc.py build/ipc-roundtrip
"""

import datetime
import os
import subprocess
import sys
import tempfile
from decimal import Decimal

import pyarrow as pa
import pyarrow.ipc as ipc

TOOL = sys.argv[1] if len(sys.argv) > 1 else "build/ipc-roundtrip"
failures = 0


def run(*args):
    out = subprocess.run([TOOL, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{out.stdout}{out.stderr}")
    return out.stdout.strip()


def check(name, fn):
    global failures
    try:
        fn()
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001 - report every case
        failures += 1
        print(f"FAIL {name}: {e}")


def write_stream(path, batches, options=None):
    with pa.OSFile(path, "wb") as sink:
        with ipc.new_stream(sink, batches[0].schema, options=options) as w:
            for b in batches:
                w.write_batch(b)


def read_stream(path):
    with pa.memory_map(path) as source:
        return ipc.open_stream(source).read_all()


def read_batches(path):
    """Every batch, empty ones included, which `read_all` drops."""
    with pa.memory_map(path) as source:
        return list(ipc.open_stream(source))


tmp = tempfile.mkdtemp(prefix="arrow-mlake-ipc-")
p = lambda name: os.path.join(tmp, name)  # noqa: E731


def mojo_to_pyarrow():
    run("write", p("mojo.arrows"))
    t = read_stream(p("mojo.arrows"))
    t.validate(full=True)
    assert t.num_rows == 11, t.num_rows
    assert t.num_columns == 16, t.num_columns
    assert t.column("i64").to_pylist()[:4] == [-5, -4, None, -2]
    assert t.column("s").to_pylist()[:5] == ["", "value-1", "value-2", None, "value-4"]
    assert t.column("dec").to_pylist()[0] == Decimal("0.000001000")
    assert t.column("d").to_pylist()[0] == datetime.date(2017, 11, 16)
    assert t.column("nul").null_count == 11
    assert t.schema.field("ts").type == pa.timestamp("us", tz="UTC")
    assert t.schema.field("m").type.key_field.nullable is False
    assert str(t.schema.field("uuid").type) in ("extension<arrow.uuid>", "uuid"), t.schema.field("uuid").type
    write_stream(p("pyarrow.arrows"), t.to_batches())
    run("check", p("pyarrow.arrows"))


def every_type_batches():
    n = 40
    ints = [None if i % 7 == 3 else i - 20 for i in range(n)]
    strs = [None if i % 5 == 2 else "x" * (i % 9) + str(i) for i in range(n)]
    cols = {
        "i8": pa.array(ints, pa.int8()),
        "i16": pa.array(ints, pa.int16()),
        "i32": pa.array(ints, pa.int32()),
        "i64": pa.array(ints, pa.int64()),
        "u8": pa.array([None if v is None else abs(v) for v in ints], pa.uint8()),
        "u16": pa.array([None if v is None else abs(v) for v in ints], pa.uint16()),
        "u32": pa.array([None if v is None else abs(v) for v in ints], pa.uint32()),
        "u64": pa.array([None if v is None else abs(v) for v in ints], pa.uint64()),
        "f16": pa.array([None if v is None else float(v) / 2 for v in ints], pa.float16()),
        "f32": pa.array([None if v is None else float(v) / 4 for v in ints], pa.float32()),
        "f64": pa.array([None if v is None else float(v) / 8 for v in ints], pa.float64()),
        "b": pa.array([None if v is None else v % 3 == 0 for v in ints], pa.bool_()),
        "s": pa.array(strs, pa.string()),
        "ls": pa.array(strs, pa.large_string()),
        "bin": pa.array([None if s is None else s.encode() + b"\xff" for s in strs], pa.binary()),
        "lbin": pa.array([None if s is None else s.encode() for s in strs], pa.large_binary()),
        "fsb": pa.array([bytes([i % 256] * 5) for i in range(n)], pa.binary(5)),
        "dec": pa.array([None if v is None else Decimal(v) / 100 for v in ints], pa.decimal128(12, 2)),
        "date": pa.array([None if v is None else datetime.date(2020, 1, 1) + datetime.timedelta(days=v) for v in ints], pa.date32()),
        "t32s": pa.array([None if v is None else abs(v) for v in ints], pa.time32("s")),
        "t32ms": pa.array([None if v is None else abs(v) * 1000 for v in ints], pa.time32("ms")),
        "t64us": pa.array([None if v is None else abs(v) * 10**6 for v in ints], pa.time64("us")),
        "t64ns": pa.array([None if v is None else abs(v) * 10**9 for v in ints], pa.time64("ns")),
        "ts_s": pa.array(ints, pa.timestamp("s")),
        "ts_ms_tz": pa.array(ints, pa.timestamp("ms", tz="America/New_York")),
        "ts_ns_utc": pa.array(ints, pa.timestamp("ns", tz="UTC")),
        "nul": pa.nulls(n),
        # An extension type: its name travels as field metadata.
        "uuid": pa.ExtensionArray.from_storage(
            pa.uuid(), pa.array([bytes([i] * 16) for i in range(n)], pa.binary(16))
        ),
        "list": pa.array([None if i % 6 == 1 else list(range(i % 4)) for i in range(n)], pa.list_(pa.int32())),
        "llist": pa.array([[str(j) for j in range(i % 3)] for i in range(n)], pa.large_list(pa.string())),
        "struct": pa.array(
            [None if i % 8 == 5 else {"a": i, "b": strs[i], "c": [i, None]} for i in range(n)],
            pa.struct([("a", pa.int64()), ("b", pa.string()), ("c", pa.list_(pa.int16()))]),
        ),
        "map": pa.array(
            [None if i % 9 == 4 else [(f"k{j}", j * i) for j in range(i % 3)] for i in range(n)],
            pa.map_(pa.string(), pa.int64()),
        ),
        "deep": pa.array(
            [[{"x": [[i, i + 1], []]}] if i % 2 else [] for i in range(n)],
            pa.list_(pa.struct([("x", pa.list_(pa.list_(pa.int64())))])),
        ),
    }
    schema = pa.schema(
        [pa.field(k, v.type, nullable=(k != "fsb"), metadata={"PARQUET:field_id": str(i)} if i % 2 else None)
         for i, (k, v) in enumerate(cols.items())]
    )
    full = pa.RecordBatch.from_arrays(list(cols.values()), schema=schema)
    # Sliced batches: pyarrow writes their buffers rebased, but a slice's
    # bitmap no longer starts on a byte boundary in memory.
    return [full.slice(0, 13), full.slice(13, 17), full.slice(30), full.slice(0, 0)]


def pyarrow_to_mojo():
    batches = every_type_batches()
    write_stream(p("every.arrows"), batches)
    out = run("copy", p("every.arrows"), p("every.back.arrows"))
    assert out.startswith(f"copied {len(batches)} batches"), out
    got = read_batches(p("every.back.arrows"))
    for b in got:
        b.validate(full=True)
    assert got[0].schema.equals(batches[0].schema, check_metadata=True), (
        f"schema changed:\n{batches[0].schema}\nbecame\n{got[0].schema}"
    )
    assert len(got) == len(batches), len(got)
    for i, (want, have) in enumerate(zip(batches, got)):
        for name in want.schema.names:
            a, b = want.column(name), have.column(name)
            assert a.equals(b), f"batch {i} column {name}:\n{a}\nbecame\n{b}"


def compressed_is_refused():
    batches = every_type_batches()[:1]
    write_stream(p("zstd.arrows"), batches, ipc.IpcWriteOptions(compression="zstd"))
    try:
        run("copy", p("zstd.arrows"), p("zstd.back.arrows"))
    except RuntimeError as e:
        assert "compressed" in str(e), e
        return
    raise AssertionError("a compressed stream was accepted")


def dictionary_is_refused():
    t = pa.table({"d": pa.array(["a", "b", "a"]).dictionary_encode()})
    write_stream(p("dict.arrows"), t.to_batches())
    try:
        run("copy", p("dict.arrows"), p("dict.back.arrows"))
    except RuntimeError as e:
        assert "dictionary" in str(e), e
        return
    raise AssertionError("a dictionary-encoded stream was accepted")


print("pyarrow", pa.__version__)
check("Mojo writes the corpus; pyarrow reads it and writes it back; Mojo checks", mojo_to_pyarrow)
check("pyarrow writes every type, sliced; Mojo copies it; pyarrow compares", pyarrow_to_mojo)
check("a compressed stream is refused by name", compressed_is_refused)
check("a dictionary-encoded stream is refused by name", dictionary_is_refused)
sys.exit(1 if failures else 0)
