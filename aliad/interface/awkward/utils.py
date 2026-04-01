from __future__ import annotations

from typing import Union

import numpy as np
import awkward as ak
from awkward import contents as akc

ShapeDim = Union[int, None]


def get_shape(array: ak.Array) -> tuple[ShapeDim, ...]:
    """Extract the shape of an awkward array, using None for variable-length axes.

    Walks the layout (content) tree of the array, inspecting both the
    structural type of each node and the actual data lengths to determine
    the shape.

    *  ``RegularArray`` and ``NumpyArray`` with ``inner_shape`` contribute
       their fixed sizes.
    *  ``ListOffsetArray`` / ``ListArray`` contribute their uniform inner
       length if every sublist has the same size, otherwise ``None``.
    *  ``RecordArray`` contributes the number of fields as a fixed
       dimension.  The walk then continues *into* the field sub-layouts
       as long as all fields share the same shape at each subsequent
       level.  When the field sub-shapes diverge (or a nested
       ``RecordArray`` is encountered), ``None`` is appended and the walk
       stops.
    *  Option-type layouts (``IndexedOptionArray``, ``ByteMaskedArray``,
       ``BitMaskedArray``, ``UnmaskedArray``) are transparent — the walk
       continues into their content without adding a dimension.
    *  ``UnionArray`` is transparent — the walk descends into the first
       variant.

    Parameters
    ----------
    array : ak.Array
        The awkward array whose shape to inspect.

    Returns
    -------
    tuple[int | None, ...]
        A tuple where each element is either an ``int`` (for axes whose
        size is constant across the array) or ``None`` (for axes whose
        size varies).

    Examples
    --------
    >>> import numpy as np
    >>> get_ak_shape(ak.Array([1, 2, 3]))
    (3,)

    >>> get_ak_shape(ak.Array([[1, 2], [3, 4, 5]]))
    (2, None)

    >>> get_ak_shape(ak.from_numpy(np.zeros((4, 3))))
    (4, 3)

    Scalar record fields contribute the field count as the last dimension:

    >>> get_ak_shape(ak.Array([{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]))
    (2, 2)

    When all record fields have uniformly-sized sublists, the size is
    resolved from the data:

    >>> arr = ak.Array([{"a": [1, 2, 3], "b": [4, 5, 6]}])
    >>> get_ak_shape(arr)
    (1, 2, 3)
    """
    if not isinstance(array.type, ak.types.ArrayType):
        raise TypeError(
            f"Expected an ak.Array with ArrayType, got {type(array.type).__name__}"
        )

    dims: list[ShapeDim] = [len(array)]
    _collect_dims_from_layout(array.layout, dims)
    return tuple(dims)


# ---------------------------------------------------------------------------
# Main layout walker
# ---------------------------------------------------------------------------

# Option-type layouts that are transparent (no dimension added).
_OPTION_TYPES = (
    akc.IndexedOptionArray,
    akc.ByteMaskedArray,
    akc.BitMaskedArray,
    akc.UnmaskedArray,
)


def _collect_dims_from_layout(layout: akc.Content, dims: list[ShapeDim]) -> None:
    """Recursively walk a layout tree and append dimension entries to *dims*.

    Parameters
    ----------
    layout : ak.contents.Content
        The current layout node being inspected.
    dims : list[int | None]
        Accumulator list that dimension entries are appended to.
    """
    if isinstance(layout, akc.RegularArray):
        dims.append(layout.size)
        _collect_dims_from_layout(layout.content, dims)

    elif isinstance(layout, akc.NumpyArray):
        # NumpyArray may carry extra dimensions in inner_shape (e.g. a
        # 2-D NumPy array ingested without regulararray=True).
        for s in layout.inner_shape:
            dims.append(s)

    elif isinstance(layout, (akc.ListOffsetArray, akc.ListArray)):
        dim = _uniform_list_size(layout)
        dims.append(dim)
        _collect_dims_from_layout(layout.content, dims)

    elif isinstance(layout, akc.RecordArray):
        dims.append(len(layout.fields))
        if layout.contents:
            _collect_record_subdims_from_layout(list(layout.contents), dims)

    elif isinstance(layout, _OPTION_TYPES):
        _collect_dims_from_layout(layout.content, dims)

    elif isinstance(layout, akc.UnionArray):
        if layout.contents:
            _collect_dims_from_layout(layout.contents[0], dims)

    elif isinstance(layout, akc.IndexedArray):
        _collect_dims_from_layout(layout.content, dims)

    elif isinstance(layout, akc.EmptyArray):
        pass

    else:
        # Future-proof: unknown layout type; stop gracefully.
        pass


# ---------------------------------------------------------------------------
# Uniform list-length detection
# ---------------------------------------------------------------------------


def _uniform_list_size(
    layout: akc.ListOffsetArray | akc.ListArray,
) -> ShapeDim:
    """Return the uniform sublist length if all sublists match, else None.

    Parameters
    ----------
    layout : ListOffsetArray | ListArray
        A list-type layout node.

    Returns
    -------
    int | None
        The common length if every sublist has the same size, otherwise
        ``None``.
    """
    starts = np.asarray(layout.starts)
    stops = np.asarray(layout.stops)

    if len(starts) == 0:
        return None

    lengths = stops - starts
    first = int(lengths[0])

    if np.all(lengths == first):
        return first

    return None


# ---------------------------------------------------------------------------
# Record sub-shape tracing (layout-based)
# ---------------------------------------------------------------------------


def _collect_record_subdims_from_layout(
    field_layouts: list[akc.Content],
    dims: list[ShapeDim],
) -> None:
    """Compare the sub-shapes of all record field layouts and extend *dims*.

    At each level the function peels one dimension layer off every field
    layout.  If all fields are leaves, nothing is appended and recursion
    stops.  If all fields agree on the same dimension value, that value
    is appended and the function recurses into the remaining inner
    layouts.  On any disagreement ``None`` is appended and recursion
    stops.

    Parameters
    ----------
    field_layouts : list[ak.contents.Content]
        The ``.contents`` of the ``RecordArray``.
    dims : list[int | None]
        Accumulator list that dimension entries are appended to.
    """
    layers = [_peel_one_layout_layer(fl) for fl in field_layouts]

    # All fields are leaves -> nothing more to trace.
    if all(inner is None for _, inner in layers):
        return

    # Check that every field agrees on the dimension value.
    dim_values = [dim for dim, _ in layers]
    first_dim = dim_values[0]

    if any(d != first_dim for d in dim_values):
        dims.append(None)
        return

    # Some fields are leaves (inner is None) while others have content —
    # structural mismatch even if the dim values coincidentally agree.
    inners = [inner for _, inner in layers]
    if any(inner is None for inner in inners):
        dims.append(None)
        return

    # All fields agree and all have further content.
    dims.append(first_dim)
    _collect_record_subdims_from_layout(inners, dims)  # type: ignore[arg-type]


def _peel_one_layout_layer(
    layout: akc.Content,
) -> tuple[ShapeDim, akc.Content | None]:
    """Peel one dimension layer off a layout, returning (dim, inner_layout).

    Parameters
    ----------
    layout : ak.contents.Content
        The layout node to inspect.

    Returns
    -------
    tuple[int | None, ak.contents.Content | None]
        A ``(dim, inner)`` pair.  ``dim`` is the dimension contributed by
        this layer.  ``inner`` is the remaining layout to recurse into,
        or ``None`` if this is a leaf / stopping point.
    """
    # Unwrap option-type and indexed wrappers transparently.
    if isinstance(layout, (*_OPTION_TYPES, akc.IndexedArray)):
        return _peel_one_layout_layer(layout.content)

    if isinstance(layout, akc.RegularArray):
        return (layout.size, layout.content)

    if isinstance(layout, (akc.ListOffsetArray, akc.ListArray)):
        dim = _uniform_list_size(layout)
        return (dim, layout.content)

    if isinstance(layout, akc.NumpyArray) and layout.inner_shape:
        return (layout.inner_shape[0], None)

    # Leaf types (NumpyArray without inner_shape, EmptyArray) and
    # nested RecordArray → stop.
    return (None, None)

# ---------------------------------------------------------------------------
# normalize_axis
# ---------------------------------------------------------------------------

def normalize_axis(array: Union[ak.Array, ak.Record], axis: int) -> Union[int, None]:
    """
    Normalize a potentially negative axis into a positive integer.

    For non-branching structures, this converts negative indices to positive 
    indices based on the array depth. For branching (ragged) RecordArrays, 
    negative axes are ambiguous and will return None.

    Parameters
    ----------
    array : ak.Array or ak.Record
        The data structure to resolve the axis against.
    axis : int
        The axis index to normalize. Can be negative.

    Returns
    -------
    int or None
        The resolved positive axis index, or None if the structure is 
        branching and the input axis is negative.

    Raises
    ------
    ValueError
        If axis is 0 for an ak.Record (which is a scalar).
        If a negative axis is out of bounds for the array depth.
    """
    if isinstance(array, ak.Record):
        if axis == 0:
            raise ValueError("Record type at axis=0 is a scalar, not an array.")
        return resolve_axis(array.array, axis)

    if axis >= 0:
        return axis

    is_branching, max_depth = array.layout.branch_depth

    if not is_branching:
        pos_axis = axis + max_depth
        if pos_axis < 0:
            raise ValueError(f"axis {axis} is out of bounds for depth {max_depth}")
        return pos_axis
    else:
        return None

# ---------------------------------------------------------------------------
# Layout navigation helpers
# ---------------------------------------------------------------------------
 
 
def _unwrap_layout(layout: akc.Content) -> akc.Content:
    """Strip option-type and indexed wrappers from a layout node.
 
    Parameters
    ----------
    layout : ak.contents.Content
        A layout node that may be wrapped in option or index layers.
 
    Returns
    -------
    ak.contents.Content
        The innermost non-wrapper layout.
    """
    while isinstance(layout, (*_OPTION_TYPES, akc.IndexedArray)):
        layout = layout.content
    return layout
 
 
def _layout_at_axis(layout: akc.Content, axis: int) -> tuple[akc.Content, int]:
    """Navigate a layout tree *axis* list-levels deep.
 
    At each level the function steps through one list or regular
    dimension, unwrapping any option / index wrappers encountered along
    the way.  When a ``NumpyArray`` with ``inner_shape`` is reached
    before all axis levels have been consumed, the remaining levels are
    tracked as *skipped inner dimensions* — these must be stripped from
    the front of ``inner_shape`` when building fillers.
 
    Parameters
    ----------
    layout : ak.contents.Content
        The root layout to navigate from.
    axis : int
        The number of list-levels to descend.  Must be non-negative.
 
    Returns
    -------
    tuple[ak.contents.Content, int]
        The layout node at the requested depth, and the number of
        ``NumpyArray.inner_shape`` dimensions that were consumed during
        navigation (0 in most cases).
    """
    node = _unwrap_layout(layout)
    inner_dims_consumed = 0
    for _ in range(axis):
        if isinstance(node, (akc.ListOffsetArray, akc.ListArray, akc.RegularArray)):
            node = _unwrap_layout(node.content)
        elif isinstance(node, akc.NumpyArray) and node.inner_shape:
            # The remaining axes live inside inner_shape — count them
            # but we can't actually step into a NumpyArray.
            inner_dims_consumed += 1
        else:
            break
    return node, inner_dims_consumed
 
 
# ---------------------------------------------------------------------------
# Filler construction
# ---------------------------------------------------------------------------
 
 
def _get_leaf_shape_and_dtype(
    layout: akc.Content,
    skip_inner_dims: int = 0,
) -> tuple[tuple[int, ...], np.dtype | None]:
    """Walk a non-record layout subtree to its leaf and return (shape, dtype).
 
    Collects dimension sizes from ``RegularArray``, ``ListOffsetArray``
    (uniform only), ``ListArray`` (uniform only), and ``NumpyArray``
    inner shapes.
 
    Parameters
    ----------
    layout : ak.contents.Content
        A layout subtree rooted at a single record field.
    skip_inner_dims : int, optional
        Number of leading ``NumpyArray.inner_shape`` dimensions to skip.
        Used when ``_layout_at_axis`` consumed axes that live inside
        ``inner_shape``.
 
    Returns
    -------
    tuple[tuple[int, ...], numpy.dtype | None]
        The accumulated shape and the leaf ``dtype``.  If the subtree
        contains a truly variable-length list or an empty array, the
        shape is truncated at that point and ``dtype`` may be ``None``.
    """
    layout = _unwrap_layout(layout)
    dims: list[int] = []
 
    while True:
        if isinstance(layout, akc.NumpyArray):
            dims.extend(layout.inner_shape[skip_inner_dims:])
            return tuple(dims), layout.dtype
 
        if isinstance(layout, akc.RegularArray):
            dims.append(layout.size)
            layout = _unwrap_layout(layout.content)
            continue
 
        if isinstance(layout, (akc.ListOffsetArray, akc.ListArray)):
            size = _uniform_list_size(layout)
            if size is not None:
                dims.append(size)
                layout = _unwrap_layout(layout.content)
                continue
            # Truly variable → can't determine shape.
            return tuple(dims), None
 
        # EmptyArray, RecordArray, UnionArray, etc. → stop.
        return tuple(dims), None
 
 
def build_filler(
    layout: akc.Content,
    value: object = 0,
    skip_inner_dims: int = 0,
) -> object:
    """Build a fill value whose structure matches the layout at a given depth.
 
    For ``RecordArray`` nodes a ``dict`` is returned with one entry per
    field, each recursively built to match that field's subtree.  For
    non-record nodes a scalar or ``numpy.ndarray`` is returned with the
    correct shape and dtype.
 
    Parameters
    ----------
    layout : ak.contents.Content
        The layout node describing the structure to fill.  Typically
        obtained from :func:`_layout_at_axis`.
    value : scalar, optional
        The fill value.  Default is ``0``.
    skip_inner_dims : int, optional
        Number of leading ``NumpyArray.inner_shape`` dimensions to skip.
        Passed through from :func:`_layout_at_axis` when the target axis
        lives inside a ``NumpyArray``'s packed dimensions.
 
    Returns
    -------
    dict | numpy.ndarray | scalar
        A structure that can be passed to :func:`ak.fill_none`.
    """
    layout = _unwrap_layout(layout)
 
    if isinstance(layout, akc.RecordArray):
        result: dict[str, object] = {}
        for field, content in zip(layout.fields, layout.contents):
            result[field] = build_filler(content, value)
        return result
 
    shape, dtype = _get_leaf_shape_and_dtype(layout, skip_inner_dims)
    if dtype is None:
        return value
    if shape:
        return np.full(shape, value, dtype=dtype)
    return dtype.type(value)
 
 
# ---------------------------------------------------------------------------
# pad_and_fill
# ---------------------------------------------------------------------------
 
 
def pad_and_fill(
    array: ak.Array,
    pad_size: int,
    value: object = 0,
    axis: int = 1,
    clip: bool = False,
) -> ak.Array:
    """Pad a jagged array to a fixed size and fill new slots with a value.
 
    Uses :func:`get_ak_shape` to understand the array structure beyond
    the padded axis, then constructs a type-aware filler value that
    respects record field names, per-field dtypes, and inner dimension
    sizes.
 
    Parameters
    ----------
    array : ak.Array
        The input jagged or nested array to be padded.
    pad_size : int
        The target length for the specified axis.
    value : scalar, optional
        The scalar used to fill padded slots.  For record arrays the
        value is broadcast into each field with the field's own dtype.
        Default is ``0``.
    axis : int, optional
        The axis along which to apply padding.  Supports negative
        indexing.  Default is ``1``.
    clip : bool, optional
        If ``True``, truncate the array at *pad_size*.  If ``False``,
        sublists longer than *pad_size* are left unchanged.
        Default is ``False``.
 
    Returns
    -------
    ak.Array
        A padded array with ``None`` values replaced by the
        structurally-correct fill value.
 
    Examples
    --------
    >>> pad_and_fill(ak.Array([[1, 2], [3]]), pad_size=3, value=-1)
    <Array [[1, 2, -1], [3, -1, -1]] type='2 * var * int64'>
 
    >>> arr = ak.Array([[{"x": 1.0}, {"x": 2.0}], [{"x": 3.0}]])
    >>> pad_and_fill(arr, pad_size=3).to_list()
    [[{'x': 1.0}, {'x': 2.0}, {'x': 0.0}], [{'x': 3.0}, {'x': 0.0}, {'x': 0.0}]]
    """
    padded = ak.pad_none(array, pad_size, axis=axis, clip=clip)
 
    pos_axis = normalize_axis(array, axis)
 
    # Navigate the layout to the depth *at* the padded axis to discover
    # the structure that each filled slot must match.
    target_layout, skip_inner = _layout_at_axis(array.layout, pos_axis)
    filler = build_filler(target_layout, value, skip_inner)

    return ak.fill_none(padded, filler, axis=axis)

def stack(*arrays, axis:int=-1):
    expanded = [a[..., np.newaxis] for a in arrays]
    return ak.concatenate(expanded, axis=axis)

def get_record_outer_shapes(array):
    nums = ak.num(array).to_list()
    return list(nums.keys()), list(nums.values())