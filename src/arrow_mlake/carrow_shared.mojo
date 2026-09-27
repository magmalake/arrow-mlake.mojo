"""Publish a batch into a shared mapping, for a consumer in another process.

`carrow.mojo` exports over the C Data Interface, which hands a consumer
**pointers**. That is why it cannot cross a process: a pointer means nothing
in another address space. This writes the same buffers — the same Arrow memory
layout, byte for byte — into a `MappedRegion`, and describes where each one
landed in a small JSON manifest.

A consumer maps the file, adds its own base to every offset, and wraps the
buffers where they lie. Nothing is decoded and nothing is copied on the
reading side:

```python
base, _ = map_the_file(path)
buf = pa.foreign_buffer(base + off, length, owner)
col = pa.Array.from_buffers(pa.float64(), n, [validity, buf], null_count=k)
```

## What it costs, honestly

One copy, on the producer's side: the decode wrote its buffers somewhere, and
this moves them into the mapping. That is the same copy `export_c` already
makes for an in-process handover — so this is a *cross-process* handover at
the price of the in-process one, not a free lunch.

Removing that copy as well means the decode allocating into the region in the
first place, which is a change to `ArrayData`'s buffers rather than to this
file. `memory_region` is the half of that which already exists.

## Flat columns only, for now

Primitive, `utf8` and `binary` columns: validity, offsets and values. Nested
types — struct, list, map — have children, and a manifest that describes a
tree is a bigger thing than the one here. `export_shared` raises on them
rather than writing something a consumer would misread.
"""

from memory_region import BumpAllocator, MappedRegion

from arrow_mlake.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_LARGE_BINARY,
    AT_LARGE_UTF8,
    AT_UTF8,
    ArrayArena,
    ArrayData,
)
from arrow_mlake.carrow import (
    _align8,
    _offsets_bytes,
    _validity_bytes,
    _values_bytes,
    n_buffers_for,
)
from std.memory import unsafe_memcpy


def _json_escape(s: String) -> String:
    """Column names are not always tame; the manifest is JSON."""
    var out = String("")
    for cp in s.codepoint_slices():
        if cp == '"':
            out += '\\"'
        elif cp == "\\":
            out += "\\\\"
        else:
            out += String(cp)
    return out^


def shared_size(arena: ArrayArena, roots: List[Int]) raises -> Int:
    """How many bytes publishing these columns will take.

    A region cannot grow, so a caller publishing several batches into one has
    to add this up first. Every buffer is padded to 8 because a consumer casts
    them where they lie.
    """
    var size = 32
    for i in range(len(roots)):
        ref a = arena.nodes[roots[i]]
        if len(a.children):
            raise Error(
                "carrow_shared: '"
                + a.name
                + "' is a nested column, which this does not publish yet"
            )
        size += _align8(_validity_bytes(a)) + 8
        size += _align8(_offsets_bytes(a)) + 8
        size += _align8(_values_bytes(a)) + 8
    return size


def export_shared_into(
    mut bump: BumpAllocator[MappedRegion],
    arena: ArrayArena,
    roots: List[Int],
    names: List[String],
) raises -> String:
    """Write these columns into a region already open, and describe them.

    The offsets in the manifest are from the region's base, not from this
    batch, so several batches can share one mapping: the producer sizes it for
    all of them, writes each in turn, and publishes each as it lands. That is
    what keeps a mapping's create/size/map/unmap cycle from being paid per
    batch — it was the larger half of publishing, bigger than the copy.
    """
    if len(roots) != len(names):
        raise Error("carrow_shared: a name per column, please")

    var manifest = String('{"columns":[')
    for i in range(len(roots)):
        ref a = arena.nodes[roots[i]]
        if len(a.children):
            raise Error(
                "carrow_shared: '"
                + a.name
                + "' is a nested column, which this does not publish yet"
            )
        if i:
            manifest += ","
        manifest += '{"name":"' + _json_escape(names[i]) + '"'
        manifest += ',"format":"' + a.type.format() + '"'
        manifest += ',"length":' + String(a.length)
        manifest += ',"null_count":' + String(a.null_count)
        manifest += ',"buffers":['

        var nbuf = n_buffers_for(a)
        var slot = 0

        # buffer 0: validity, absent when nothing is null
        if nbuf > 0:
            if a.null_count > 0:
                manifest += _put(bump, Span(a.validity), _validity_bytes(a))
            else:
                manifest += "null"
            slot = 1

        var ob = _offsets_bytes(a)
        if ob > 0 and slot < nbuf:
            var wide = ob == 8 * (a.length + 1)
            # The offsets lists are native-endian already, and so is every
            # Arrow buffer, so they go over as their own bytes.
            if wide:
                manifest += "," + _put_raw(
                    bump,
                    Int(a.large_offsets.unsafe_ptr()),
                    8 * len(a.large_offsets),
                    ob,
                )
            else:
                manifest += "," + _put_raw(
                    bump, Int(a.offsets.unsafe_ptr()), 4 * len(a.offsets), ob
                )
            slot += 1

        var vb = _values_bytes(a)
        if slot < nbuf:
            manifest += "," + _put(bump, Span(a.values), vb)
            slot += 1

        manifest += "]}"

    manifest += '],"bytes":' + String(bump.used) + "}"
    return manifest^


def export_shared(
    arena: ArrayArena, roots: List[Int], names: List[String], path: String
) raises -> String:
    """One batch into a mapping of its own; returns the manifest.

    The convenience form of `export_shared_into` for a caller with a single
    batch to publish. The region is unmapped before returning: the producer is
    finished with it, and the file carries the bytes.
    """
    var bump = BumpAllocator[MappedRegion](
        MappedRegion(path, shared_size(arena, roots))
    )
    var manifest = export_shared_into(bump, arena, roots, names)
    bump^.close()
    return manifest^


def _put_raw(
    mut bump: BumpAllocator[MappedRegion], src: Int, have: Int, want: Int
) raises -> String:
    """`_put` for a buffer whose elements are not bytes, by address."""
    if want <= 0:
        return "null"
    var at = bump.claim(want)
    var n = want if want < have else have
    if n:
        unsafe_memcpy(
            dest=bump.unsafe_ptr(at),
            src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src),
            count=n,
        )
    return '{"offset":' + String(at) + ',"length":' + String(want) + "}"


def _put(
    mut bump: BumpAllocator[MappedRegion], src: Span[UInt8, _], want: Int
) raises -> String:
    """Copy one buffer into the region; return its manifest entry.

    A source shorter than the buffer Arrow expects leaves zeros behind it,
    which is what a freshly mapped file gives for free.
    """
    if want <= 0:
        return "null"
    # `append` copies and says where; the buffer Arrow expects may be longer
    # than the source, and the mapping is already zeroed behind it.
    var at = bump.append(src[: want if want < len(src) else len(src)])
    if want > len(src):
        _ = bump.claim(want - len(src))
    return '{"offset":' + String(at) + ',"length":' + String(want) + "}"
