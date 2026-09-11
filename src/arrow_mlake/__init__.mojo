"""`arrow-mlake.mojo` — the Arrow memory layout and the C Data Interface, in
pure Mojo.

Part of magmalake: data lake building blocks in Mojo.

```mojo
from arrow_mlake import ImportedStream

var s = ImportedStream(stream_address)   # a producer's ArrowArrayStream
while True:
    var got = s.next()
    if not got:
        break
    ref batch = got.value()
    print(batch.num_rows, "rows,", batch.num_columns(), "columns")
```

Five modules, and no dependencies on anything but `std`:

* `arrow_mlake.arrow` — `ArrowType`, `ArrayData` and `ArrayArena`: one Arrow
  array as the columnar spec lays it out, with nesting held in an arena
  because Mojo does not allow a struct to contain a `List` of itself.
* `arrow_mlake.carrow` — the C Data Interface outbound. `export_c` turns an
  array into an `ArrowSchema`/`ArrowArray` pair any Arrow implementation can
  consume.
* `arrow_mlake.carrow_import` — the same interface inbound, including
  `ArrowArrayStream`, which is the shape a scan arrives in.
* `arrow_mlake.carrow_stream` — `export_stream`, the outbound half of that:
  a sequence of batches as one `ArrowArrayStream` a consumer pulls from.
* `arrow_mlake.batch` — `RecordBatch`, a run of rows as one array per column.

**This tin is a stopgap.** The community Arrow implementation for Mojo is
[marrow](https://github.com/kszucs/marrow), which is more complete than this
and which this should be replaced by as soon as it builds on Mojo 1.x. See the
README.
"""

from arrow_mlake.arrow import (
    AT_BINARY,
    AT_BOOL,
    AT_DATE32,
    AT_DECIMAL128,
    AT_FIXED_SIZE_BINARY,
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
    AT_TIME32,
    AT_TIME64,
    AT_TIMESTAMP,
    AT_UINT8,
    AT_UINT16,
    AT_UINT32,
    AT_UINT64,
    AT_UTF8,
    TU_MICRO,
    TU_MILLI,
    TU_NANO,
    TU_SECOND,
    ArrayArena,
    ArrayData,
    ArrowType,
    at,
    at_decimal,
    at_fixed,
    at_time,
    at_timestamp,
    bit_fill_valid,
    bit_get,
    bit_set,
    bitmap_bytes,
    load_f32,
    load_f64,
    load_i32,
    load_i64,
    load_u64,
    store_u32,
    store_u64,
    unit_name,
    unit_suffix,
)
from arrow_mlake.batch import (
    RecordBatch,
    array_bool,
    array_bool_into,
    array_f64,
    array_f64_into,
    array_i64,
    array_i64_into,
    array_str,
    array_str_into,
)
from arrow_mlake.carrow import (
    ARROW_FLAG_NULLABLE,
    CArrowArray,
    CArrowSchema,
    ExportedArray,
    export_c,
    n_buffers_for,
    n_buffers_for_type,
)
from arrow_mlake.carrow_import import (
    MAX_IMPORT_DEPTH,
    CArrowArrayStream,
    ImportedArray,
    ImportedStream,
    extension_name,
    import_batch_c,
    import_c,
    n_children_for_type,
    parquet_field_id,
    parse_format,
    release_c_array,
    release_c_schema,
)
from arrow_mlake.carrow_stream import export_stream, export_stream_of
