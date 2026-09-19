"""Opening a region, for a caller that does not want the allocator's types.

`export_shared_into` takes a `BumpAllocator[MappedRegion]`, which a tool would
otherwise have to name — and naming it means importing `memory_region` just to
spell a parameter. This is the one line that saves every publisher from that.
"""

from memory_region import BumpAllocator, MappedRegion


def open_split_region(
    path: String, size: Int
) raises -> BumpAllocator[MappedRegion]:
    """A mapping at `path`, sized for everything about to go into it."""
    return BumpAllocator[MappedRegion](MappedRegion(path, size))
