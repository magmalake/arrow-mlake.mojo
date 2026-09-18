"""Producing an `ArrowArrayStream`: the C Data Interface, outbound, in batches.

`carrow.export_c` hands one array across the boundary. This hands a *sequence*
of them, which is the shape every consumer wants for a scan — DuckDB, pyarrow
and polars all take an `ArrowArrayStream` and pull batches until it ends.

## Why this is the local answer

Flight moves Arrow between processes. In one process there is nothing to move:
the consumer reads our buffers where they already are, and the only thing that
crosses is a struct of pointers. A DuckDB extension loading a Mojo shared
library is exactly that case, and reaching for a wire protocol there would add
an encode and a decode to a problem with neither.

## The contract, and who frees what

A stream is five fields: four C function pointers and a `private_data` the
producer owns. The consumer calls `get_next` until it receives an array whose
`release` is NULL — that, not an error, is how a stream ends — and then calls
`release` on the stream itself.

Ownership is strict and worth stating because getting it wrong is a
double-free rather than a wrong answer:

* Every array handed out becomes **the consumer's** to release. This exports
  each batch up front and then *moves* it out on `get_next`, clearing our copy
  so nothing is released twice.
* The schema is handed out the same way, once per `get_schema` call, so a
  consumer that asks twice gets two independently-releasable copies.
* Whatever the consumer never asked for is freed by `release`.

The callbacks are `abi("C")` and take raw addresses because that is what the
other side of the boundary can call. Everything they touch is heap-allocated
and outlives the Mojo frame that built it — a stream handed to C and then
pointing at a stack is the mistake this file exists to not make.
"""

from std.memory.alloc import unsafe_alloc

from arrow_mlake.arrow import ArrayArena
from arrow_mlake.carrow import (
    CArrowArray,
    CArrowSchema,
    ExportedArray,
    export_c,
)
from arrow_mlake.carrow_import import (
    CArrowArrayStream,
    StreamGetLastErrorFn,
    StreamGetNextFn,
    StreamGetSchemaFn,
    StreamReleaseFn,
    release_c_array,
    release_c_schema,
)

comptime _OK: Int32 = 0
comptime _StreamPtr = Pointer[CArrowArrayStream, MutUntrackedOrigin]


def _released_schema() -> CArrowSchema:
    """All-zero, so `release` is NULL: what a consumer reads as "nothing here".
    """
    return CArrowSchema(0, 0, 0, 0, 0, 0, 0, 0, 0)


def _released_array() -> CArrowArray:
    return CArrowArray(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


@fieldwise_init
struct _StreamState(Copyable, Movable):
    """What a live stream still owns: batches not yet handed out, and a cursor.

    Four integers and no Mojo-owned collection, deliberately: this lives in
    memory a C callback reaches through `private_data`, where a destructor
    would never run. Everything here is freed explicitly by `_release`.
    """

    var schema: Int
    """Address of a `CArrowSchema`, or 0 once moved to the consumer."""
    var arrays: Int
    """Heap array of `n` `CArrowArray` addresses, one per batch; an entry is
    zeroed when moved out so `release` frees exactly what is left."""
    var n: Int
    var pos: Int


def _state_of(
    stream: _StreamPtr,
) -> Pointer[_StreamState, MutUntrackedOrigin]:
    return Pointer[_StreamState, MutUntrackedOrigin](
        unsafe_from_address=stream[].private_data
    )


def _get_schema(
    stream: _StreamPtr, out_schema: Pointer[CArrowSchema, MutUntrackedOrigin]
) abi("C") -> Int32:
    var st = _state_of(stream)
    if st[].schema == 0:
        # Already taken. A released schema, not a stale copy: a second caller
        # gets "nothing here" rather than a struct whose release has been used.
        out_schema[] = _released_schema()
        return _OK
    # A move, not a copy: two owners of one release callback is a double free,
    # so ours is blanked as the consumer's is filled.
    var mine = Pointer[CArrowSchema, MutUntrackedOrigin](
        unsafe_from_address=st[].schema
    )
    out_schema[] = mine[].copy()
    mine[] = _released_schema()
    st[].schema = 0
    return _OK


def _get_next(
    stream: _StreamPtr, out_array: Pointer[CArrowArray, MutUntrackedOrigin]
) abi("C") -> Int32:
    var st = _state_of(stream)
    if st[].pos >= st[].n:
        # End of stream is a released array, not an error code — a NULL
        # `release` is what the consumer checks for.
        out_array[] = _released_array()
        return _OK
    var slot = Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=st[].arrays + st[].pos * 8
    )
    var mine = Pointer[CArrowArray, MutUntrackedOrigin](
        unsafe_from_address=slot[]
    )
    out_array[] = mine[].copy()
    mine[] = _released_array()
    slot[] = 0
    st[].pos += 1
    return _OK


def _get_last_error(stream: _StreamPtr) abi("C") -> Int:
    """NULL: every failure this producer can have is raised before the stream
    exists, so a message here would describe something that cannot happen."""
    return 0


def _release(stream: _StreamPtr) abi("C") -> None:
    """Free whatever the consumer never took, then mark the stream spent."""
    var st = _state_of(stream)

    # `release_c_*` frees the whole export block through `private_data`; the
    # struct itself lives inside that block, so there is nothing else to free.
    release_c_schema(st[].schema)
    for i in range(st[].pos, st[].n):
        release_c_array(
            Pointer[Int, MutUntrackedOrigin](
                unsafe_from_address=st[].arrays + i * 8
            )[]
        )

    Pointer[Int, MutUntrackedOrigin](
        unsafe_from_address=st[].arrays
    ).unsafe_free()
    st.unsafe_free()
    stream[].private_data = 0

    # A NULL release is how the consumer knows the stream is spent, so it is
    # the last thing written.
    stream[].release = 0


def export_stream_of(var exported: List[ExportedArray]) raises -> Int:
    """Turn already-exported arrays into a stream; returns its address.

    The general form. Each `ExportedArray` has copied its buffers out of its
    arena already, so the batches need not come from the same arena — which is
    what a scan produces, one arena per batch. Ownership passes here: the
    stream releases everything the consumer does not take.

    Every array is exported before the consumer sees the first one. A lazier
    stream that read inside `get_next` would hold one batch instead of all of
    them; that needs a scan that can be resumed from inside a C callback, and
    this Mojo has no way to keep such an object alive across the boundary.
    """
    if len(exported) == 0:
        raise Error("arrow_mlake.carrow: a stream needs at least one array")

    var n = len(exported)
    var arrays = unsafe_alloc[Int](n)
    var schema = 0
    for i in range(n):
        var pair = exported[i].into_raw()
        arrays[unsafe_offset=i] = pair[0]
        if i == 0:
            # One schema for the stream: the C Data Interface requires every
            # batch to share it, so the rest are duplicates.
            schema = pair[1]
        else:
            release_c_schema(pair[1])

    var state = unsafe_alloc[_StreamState](1)
    state[] = _StreamState(schema, Int(arrays), n, 0)

    var stream = unsafe_alloc[CArrowArrayStream](1)
    stream[] = CArrowArrayStream(0, 0, 0, 0, Int(state))

    # Typed slots rather than integers: the fields are declared `Int` because
    # that is what a C pointer is, but a function is not convertible to one,
    # and writing through a typed pointer keeps the signature checked.
    var base = Int(stream)
    Pointer[StreamGetSchemaFn, MutUntrackedOrigin](
        unsafe_from_address=base
    )[] = _get_schema
    Pointer[StreamGetNextFn, MutUntrackedOrigin](
        unsafe_from_address=base + 8
    )[] = _get_next
    Pointer[StreamGetLastErrorFn, MutUntrackedOrigin](
        unsafe_from_address=base + 16
    )[] = _get_last_error
    Pointer[StreamReleaseFn, MutUntrackedOrigin](
        unsafe_from_address=base + 24
    )[] = _release
    return base


def export_stream(arena: ArrayArena, roots: List[Int]) raises -> Int:
    """Export one arena's arrays as a stream; returns its address.

    The caller hands the address on and then forgets it: the stream owns
    everything from here, and the consumer's `release` frees it — including
    the stream struct, which is why this returns an address rather than a
    value it would have to outlive.
    """
    if len(roots) == 0:
        raise Error("arrow_mlake.carrow: a stream needs at least one array")
    var exported = List[ExportedArray]()
    for i in range(len(roots)):
        exported.append(export_c(arena, roots[i]))
    return export_stream_of(exported^)
