from __future__ import annotations

import numpy as np
import awkward as ak
import pytest

from aliad.interface.awkward import get_shape, pad_and_fill, build_filler
from aliad.interface.awkward.utils import _layout_at_axis

# ---------------------------------------------------------------------------
# Flat / 1-D arrays
# ---------------------------------------------------------------------------


class TestFlat:
    """Flat (1-D) arrays should return a single-element tuple."""

    def test_integer_array(self) -> None:
        assert get_shape(ak.Array([1, 2, 3])) == (3,)

    def test_float_array(self) -> None:
        assert get_shape(ak.Array([1.1, 2.2])) == (2,)

    def test_bool_array(self) -> None:
        assert get_shape(ak.Array([True, False, True, True])) == (4,)

    def test_single_element(self) -> None:
        assert get_shape(ak.Array([42])) == (1,)

    def test_empty_array(self) -> None:
        assert get_shape(ak.Array([])) == (0,)


# ---------------------------------------------------------------------------
# Regular (fixed-size) dimensions — from NumPy
# ---------------------------------------------------------------------------


class TestRegular:
    """Arrays built from NumPy should have all-integer shapes."""

    def test_2d(self) -> None:
        arr = ak.from_numpy(np.zeros((4, 3)))
        assert get_shape(arr) == (4, 3)

    def test_3d(self) -> None:
        arr = ak.from_numpy(np.zeros((2, 3, 4)))
        assert get_shape(arr) == (2, 3, 4)

    def test_4d(self) -> None:
        arr = ak.from_numpy(np.arange(120).reshape(2, 3, 4, 5))
        assert get_shape(arr) == (2, 3, 4, 5)

    def test_regular_single_row(self) -> None:
        arr = ak.from_numpy(np.zeros((1, 7)))
        assert get_shape(arr) == (1, 7)


# ---------------------------------------------------------------------------
# Variable-length (ragged / ListType) dimensions
# ---------------------------------------------------------------------------


class TestRagged:
    """Variable-length lists: None when lengths differ, int when uniform."""

    def test_truly_ragged(self) -> None:
        arr = ak.Array([[1, 2], [3, 4, 5]])
        assert get_shape(arr) == (2, None)

    def test_uniform_var_resolved_to_int(self) -> None:
        """All sublists are the same length → resolved from data."""
        arr = ak.Array([[1, 2, 3], [4, 5, 6]])
        assert get_shape(arr) == (2, 3)

    def test_doubly_ragged(self) -> None:
        arr = ak.Array([[[1, 2], [3]], [[4, 5, 6]]])
        assert get_shape(arr) == (2, None, None)

    def test_ragged_with_empty_sublists(self) -> None:
        arr = ak.Array([[1, 2], [], [3]])
        assert get_shape(arr) == (3, None)

    def test_all_empty_sublists(self) -> None:
        """All sublists are length 0 → uniform, resolved to 0."""
        arr = ak.Array([[], [], []])
        assert get_shape(arr) == (3, 0)

    def test_deeply_nested_uniform(self) -> None:
        """Each level is uniform → each resolved to its concrete size."""
        arr = ak.Array([[[[1]]]])
        assert get_shape(arr) == (1, 1, 1, 1)

    def test_deeply_nested_truly_ragged(self) -> None:
        """Outer ragged, then inner ragged."""
        arr = ak.Array([[[1, 2], [3]], [[4]]])
        assert get_shape(arr) == (2, None, None)


# ---------------------------------------------------------------------------
# Mixed regular + ragged
# ---------------------------------------------------------------------------


class TestMixedRegularRagged:
    """The primary physics use-case: (N, None, M)."""

    def test_events_particles_features_regular(self) -> None:
        """N events, variable particles per event, M fixed features via RegularType."""
        arr = ak.Array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ]
        )
        arr = ak.to_regular(arr, axis=2)
        assert get_shape(arr) == (3, None, 3)

    def test_regular_then_ragged(self) -> None:
        """Outer regular, inner ragged."""
        arr = ak.to_regular(
            ak.Array([[[1, 2], [3]], [[4, 5], [6, 7, 8]]]),
            axis=1,
        )
        assert get_shape(arr) == (2, 2, None)

    def test_regular_ragged_regular(self) -> None:
        """Regular, ragged, then regular innermost."""
        inner = ak.Array(
            [
                [[[1, 2], [3, 4]], [[5, 6]]],
                [[[7, 8]], [[9, 10], [11, 12], [13, 14]]],
            ]
        )
        arr = ak.to_regular(inner, axis=1)
        arr = ak.to_regular(arr, axis=3)
        assert get_shape(arr) == (2, 2, None, 2)


# ---------------------------------------------------------------------------
# RecordType — field count + sub-shape tracing
# ---------------------------------------------------------------------------


class TestRecordScalarFields:
    """Records with all-scalar (NumpyType) fields."""

    def test_two_scalar_fields(self) -> None:
        arr = ak.Array([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        assert get_shape(arr) == (2, 2)

    def test_four_scalar_fields(self) -> None:
        arr = ak.Array(
            [
                {"pt": 1.0, "eta": 0.1, "phi": 0.5, "m": 0.0},
                {"pt": 2.0, "eta": -0.3, "phi": 1.2, "m": 0.1},
            ]
        )
        assert get_shape(arr) == (2, 4)

    def test_single_field(self) -> None:
        arr = ak.Array([{"x": 1}, {"x": 2}, {"x": 3}])
        assert get_shape(arr) == (3, 1)

    def test_hep_events_particles_scalar_features(self) -> None:
        """N events × variable particles × M scalar features."""
        rng = np.random.default_rng(42)
        data = []
        for _ in range(100):
            n_particles = rng.integers(0, 10)
            event = [
                {
                    "pt": rng.random(),
                    "eta": rng.random(),
                    "phi": rng.random(),
                    "m": rng.random(),
                }
                for _ in range(n_particles)
            ]
            data.append(event)
        arr = ak.Array(data)
        assert get_shape(arr) == (100, None, 4)

    def test_tuple_record(self) -> None:
        arr = ak.Array([(1, 2.0), (3, 4.0)])
        assert get_shape(arr) == (2, 2)


class TestRecordConsistentSubshape:
    """Records whose fields all share the same sub-shape trace further."""

    def test_consistent_regular_fields(self) -> None:
        """All fields are ``10 * float64`` → shape continues past the record."""
        rng = np.random.default_rng(0)
        data = []
        for _ in range(50):
            event = []
            for _ in range(rng.integers(0, 5)):
                event.append(
                    {
                        "pt": rng.random(10),
                        "eta": rng.random(10),
                        "phi": rng.random(10),
                        "m": rng.random(10),
                    }
                )
            data.append(event)
        arr = ak.to_regular(ak.Array(data), axis=-1)
        assert get_shape(arr) == (50, None, 4, 10)

    def test_consistent_var_fields_resolved_from_data(self) -> None:
        """Fields are ``var * float64`` but data is uniformly length 10.

        The layout-based walk detects uniform inner lengths and reports
        the concrete size rather than None.
        """
        rng = np.random.default_rng(0)
        data = []
        for _ in range(50):
            event = []
            for _ in range(rng.integers(0, 5)):
                event.append(
                    {
                        "pt": rng.random(10),
                        "eta": rng.random(10),
                        "phi": rng.random(10),
                        "m": rng.random(10),
                    }
                )
            data.append(event)
        arr = ak.Array(data)  # NOT regularized — type says var
        assert get_shape(arr) == (50, None, 4, 10)

    def test_consistent_2d_regular_fields(self) -> None:
        """All fields are ``2 * 2 * int64`` → two extra dims traced."""
        arr = ak.Array(
            [
                {"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]},
            ]
        )
        arr = ak.to_regular(ak.to_regular(arr, axis=-1), axis=-2)
        assert get_shape(arr) == (1, 2, 2, 2)

    def test_consistent_ragged_fields_different_lengths(self) -> None:
        """All fields are ``var * float64`` with truly different lengths → None."""
        arr = ak.Array(
            [
                {"a": [1.0, 2.0], "b": [3.0, 4.0, 5.0]},
                {"a": [6.0], "b": [7.0]},
            ]
        )
        assert get_shape(arr) == (2, 2, None)

    def test_consistent_list_fields_uniform_data(self) -> None:
        """Record fields that are var * float64 but data uniformly length 4."""
        records = [
            ak.Record(
                {
                    "a": np.random.rand(4),
                    "b": np.random.rand(4),
                    "c": np.random.rand(4),
                }
            )
            for _ in range(10)
        ]
        arr = ak.Array(records)
        # All fields have uniform length 4 → resolved from data
        assert get_shape(arr) == (10, 3, 4)


class TestRecordInconsistentSubshape:
    """Records whose fields diverge in sub-shape emit None and stop."""

    def test_mixed_regular_sizes(self) -> None:
        """Field 'pt' is 5*2*float64, others are 10*float64 → mismatch."""
        rng = np.random.default_rng(0)
        data = []
        for _ in range(50):
            event = []
            for _ in range(rng.integers(0, 5)):
                event.append(
                    {
                        "pt": rng.random(10).reshape(5, 2),
                        "eta": rng.random(10),
                        "phi": rng.random(10),
                        "m": rng.random(10),
                    }
                )
            data.append(event)
        arr = ak.to_regular(ak.Array(data), axis=-1)
        assert get_shape(arr) == (50, None, 4, None)

    def test_scalar_vs_list_field(self) -> None:
        """One field is NumpyType, another is ListType → diverge."""
        arr = ak.Array(
            [
                {"x": 1, "y": [1, 2]},
                {"x": 2, "y": [3]},
            ]
        )
        assert get_shape(arr) == (2, 2, None)

    def test_one_field_uniform_other_ragged(self) -> None:
        """Both fields are lists, but only one has uniform lengths."""
        arr = ak.Array(
            [
                {"a": [1, 2], "b": [3, 4]},
                {"a": [5, 6], "b": [7]},
            ]
        )
        # a has uniform len 2, b does not → both peeled as list, a→2, b→None → mismatch
        assert get_shape(arr) == (2, 2, None)


class TestRecordNestedRecords:
    """Nested records inside record fields."""

    def test_all_fields_are_nested_records(self) -> None:
        """Both fields are RecordArray → both peel to leaf → stop."""
        arr = ak.Array([{"a": {"x": 1}, "b": {"y": 2}}])
        assert get_shape(arr) == (1, 2)

    def test_one_field_nested_record_one_scalar(self) -> None:
        """One field is RecordArray, another is NumpyArray → both leaf → stop."""
        arr = ak.Array([{"a": {"x": 1, "y": 2}, "b": 3.0}])
        assert get_shape(arr) == (1, 2)

    def test_one_field_nested_record_one_list(self) -> None:
        """RecordArray vs ListArray → inners mismatch → None, stop."""
        arr = ak.Array([{"a": {"x": 1}, "b": [1.0, 2.0]}])
        assert get_shape(arr) == (1, 2, None)


class TestRecordEmpty:
    """Edge cases with empty records."""

    def test_empty_record(self) -> None:
        arr = ak.Array([{}, {}, {}])
        assert get_shape(arr) == (3, 0)


# ---------------------------------------------------------------------------
# OptionType — should be transparent
# ---------------------------------------------------------------------------


class TestOption:
    """OptionType layers should be skipped (no dimension added)."""

    def test_flat_with_none(self) -> None:
        arr = ak.Array([1.0, None, 2.0])
        assert get_shape(arr) == (3,)

    def test_ragged_with_none(self) -> None:
        arr = ak.Array([[1, 2], None, [3]])
        assert get_shape(arr) == (3, None)

    def test_option_inside_list(self) -> None:
        arr = ak.Array([[1, None, 2], [None]])
        assert get_shape(arr) == (2, None)

    def test_option_record(self) -> None:
        """OptionType wrapping a RecordType passes through to the record."""
        arr = ak.Array([{"x": 1.0, "y": 2.0}, None, {"x": 3.0, "y": 4.0}])
        assert get_shape(arr) == (3, 2)


# ---------------------------------------------------------------------------
# UnionType — should be transparent (follows first variant)
# ---------------------------------------------------------------------------


class TestUnion:
    """UnionType should be skipped, descending into the first variant."""

    def test_simple_union(self) -> None:
        arr = ak.concatenate(([{"x": 1}], [{"y": 2}]))
        assert get_shape(arr) == (2, 1)

    def test_numeric_union(self) -> None:
        arr = ak.concatenate(([1, 2, 3], [True, False]))
        assert get_shape(arr) == (5,)


# ---------------------------------------------------------------------------
# Data-aware uniform-length resolution
# ---------------------------------------------------------------------------


class TestUniformLengthResolution:
    """ListType arrays with uniform data lengths resolve to concrete sizes."""

    def test_uniform_var_is_resolved(self) -> None:
        """Type says var but data is uniform → concrete size."""
        arr = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert get_shape(arr) == (3, 3)

    def test_truly_ragged_stays_none(self) -> None:
        arr = ak.Array([[1, 2], [3, 4, 5]])
        assert get_shape(arr) == (2, None)

    def test_nested_uniform(self) -> None:
        """Two levels of uniform var → both resolved."""
        arr = ak.Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert get_shape(arr) == (2, 2, 2)

    def test_outer_ragged_inner_uniform(self) -> None:
        """Outer ragged, but inner is uniform within each sublist."""
        arr = ak.Array([[[1, 2], [3, 4], [5, 6]], [[7, 8]]])
        # Outer: lengths [3, 1] → None.  Inner: lengths [2, 2, 2, 2] → 2.
        assert get_shape(arr) == (2, None, 2)

    def test_string_same_length(self) -> None:
        """Strings of equal length resolve to that length (char-level)."""
        arr = ak.Array(["abc", "def", "ghi"])
        assert get_shape(arr) == (3, 3)

    def test_string_different_length(self) -> None:
        """Strings of different length → None."""
        arr = ak.Array(["hi", "hello"])
        assert get_shape(arr) == (2, None)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_type_error_on_record(self) -> None:
        """Passing an ak.Record (not ak.Array) should raise TypeError."""
        rec = ak.Array([{"x": 1}])[0]
        with pytest.raises(TypeError, match="ArrayType"):
            get_shape(rec)  # type: ignore[arg-type]

    def test_large_outer_length(self) -> None:
        arr = ak.Array(np.arange(10_000))
        assert get_shape(arr) == (10_000,)

    def test_shape_elements_are_int_or_none(self) -> None:
        """Every element of the returned tuple is int or None."""
        arr = ak.Array([[[1, 2], [3]], [[4]]])
        shape = get_shape(arr)
        for dim in shape:
            assert dim is None or isinstance(dim, int)


# ===================================================================
# Tests for pad_and_fill
# ===================================================================

# ---------------------------------------------------------------------------
# build_filler
# ---------------------------------------------------------------------------


class TestBuildFiller:
    """Unit tests for the filler construction helper."""

    def test_scalar_leaf(self) -> None:
        arr = ak.Array([[1, 2], [3]])
        layout = arr.layout.content  # NumpyArray
        filler = build_filler(layout, -1)
        assert filler == np.int64(-1)

    def test_record_scalar_fields(self) -> None:
        arr = ak.Array([{"x": 1.0, "y": 2}])
        layout = arr.layout  # RecordArray
        filler = build_filler(layout, 0)
        assert isinstance(filler, dict)
        assert set(filler.keys()) == {"x", "y"}
        assert filler["x"] == np.float64(0.0)
        assert filler["y"] == np.int64(0)

    def test_record_with_uniform_list_fields(self) -> None:
        arr = ak.Array([{"a": [1.0, 2.0], "b": [3, 4]}])
        layout = arr.layout  # RecordArray
        filler = build_filler(layout, 0)
        assert isinstance(filler, dict)
        np.testing.assert_array_equal(filler["a"], np.array([0.0, 0.0]))
        np.testing.assert_array_equal(filler["b"], np.array([0, 0]))
        assert filler["a"].dtype == np.float64
        assert filler["b"].dtype == np.int64

    def test_record_with_multidim_fields(self) -> None:
        arr = ak.Array([{"a": np.zeros((3, 2)), "b": np.ones((3, 2), dtype=np.int32)}])
        arr = ak.to_regular(ak.to_regular(arr, axis=-1), axis=-2)
        layout = arr.layout  # RecordArray
        filler = build_filler(layout, -1)
        assert filler["a"].shape == (3, 2)
        assert filler["a"].dtype == np.float64
        assert filler["b"].shape == (3, 2)
        # ak.Array promotes int32 → int64 on ingestion
        assert filler["b"].dtype == np.int64
        np.testing.assert_array_equal(filler["a"], np.full((3, 2), -1.0))
        np.testing.assert_array_equal(filler["b"], np.full((3, 2), -1, dtype=np.int64))


# ---------------------------------------------------------------------------
# pad_and_fill — basic
# ---------------------------------------------------------------------------


class TestPadAndFillBasic:
    """Basic pad_and_fill on simple ragged arrays."""

    def test_pad_simple_ragged(self) -> None:
        arr = ak.Array([[1, 2], [3, 4, 5]])
        result = pad_and_fill(arr, 4, value=-1)
        assert result.to_list() == [[1, 2, -1, -1], [3, 4, 5, -1]]

    def test_pad_no_change_needed(self) -> None:
        """Sublists already at or above pad_size (no clip)."""
        arr = ak.Array([[1, 2, 3], [4, 5, 6]])
        result = pad_and_fill(arr, 3, value=0)
        assert result.to_list() == [[1, 2, 3], [4, 5, 6]]

    def test_pad_with_clip(self) -> None:
        arr = ak.Array([[1, 2, 3, 4, 5], [6, 7]])
        result = pad_and_fill(arr, 3, value=0, clip=True)
        assert result.to_list() == [[1, 2, 3], [6, 7, 0]]

    def test_pad_empty_sublists(self) -> None:
        arr = ak.Array([[], [1]])
        result = pad_and_fill(arr, 2, value=-1)
        assert result.to_list() == [[-1, -1], [1, -1]]

    def test_pad_all_empty(self) -> None:
        arr = ak.Array([[], [], []])
        result = pad_and_fill(arr, 2, value=0)
        assert result.to_list() == [[0, 0], [0, 0], [0, 0]]


# ---------------------------------------------------------------------------
# pad_and_fill — records
# ---------------------------------------------------------------------------


class TestPadAndFillRecords:
    """pad_and_fill on arrays containing records."""

    def test_record_scalar_fields(self) -> None:
        arr = ak.Array(
            [
                [{"pt": 1.0, "eta": 0.1}, {"pt": 2.0, "eta": 0.2}],
                [{"pt": 3.0, "eta": 0.3}],
            ]
        )
        result = pad_and_fill(arr, 3, value=0)
        assert result.to_list() == [
            [{"pt": 1.0, "eta": 0.1}, {"pt": 2.0, "eta": 0.2}, {"pt": 0.0, "eta": 0.0}],
            [{"pt": 3.0, "eta": 0.3}, {"pt": 0.0, "eta": 0.0}, {"pt": 0.0, "eta": 0.0}],
        ]

    def test_record_array_fields_uniform(self) -> None:
        """Record fields that are lists of uniform length."""
        arr = ak.Array(
            [
                [{"a": [1.0, 2.0], "b": [3, 4]}, {"a": [5.0, 6.0], "b": [7, 8]}],
                [{"a": [9.0, 10.0], "b": [11, 12]}],
            ]
        )
        result = pad_and_fill(arr, 3, value=0)
        last_slot = result[1][2].to_list()
        assert last_slot == {"a": [0.0, 0.0], "b": [0, 0]}

    def test_record_preserves_field_dtypes(self) -> None:
        """Float and integer fields should keep their dtypes in the filler."""
        arr = ak.Array(
            [
                [{"x": 1.5, "y": 2}],
                [],
            ]
        )
        result = pad_and_fill(arr, 1, value=0)
        filled_slot = result[1][0]
        assert float(filled_slot["x"]) == 0.0
        assert int(filled_slot["y"]) == 0

    def test_record_with_multidim_mixed_dtype_fields(self) -> None:
        """The user's showcase: record fields with shape (5, 2), mixed dtypes."""
        rng = np.random.default_rng(42)
        data = []
        for _ in range(3):
            event = []
            for _ in range(rng.integers(1, 3)):
                event.append(
                    {
                        "a": rng.random(6).reshape(3, 2),
                        "b": rng.integers(10, size=6).reshape(3, 2),
                    }
                )
            data.append(event)
        arr = ak.to_regular(ak.to_regular(ak.Array(data), axis=-1), axis=-1)
        result = pad_and_fill(arr, 5, value=-999)
        # Check a filled slot
        filled = result[0][-1]
        assert filled["a"].to_list() == [[-999.0, -999.0]] * 3
        assert filled["b"].to_list() == [[-999, -999]] * 3

    def test_hep_events_particles_features(self) -> None:
        """Full HEP pipeline: events → variable particles → 4 scalar features."""
        rng = np.random.default_rng(0)
        data = []
        for _ in range(10):
            n = rng.integers(0, 4)
            event = [
                {
                    "pt": rng.random(),
                    "eta": rng.random(),
                    "phi": rng.random(),
                    "m": rng.random(),
                }
                for _ in range(n)
            ]
            data.append(event)
        arr = ak.Array(data)
        result = pad_and_fill(arr, 4, value=0)
        # Every event should now have exactly 4 particles
        counts = ak.num(result, axis=1)
        assert ak.all(counts >= 4)
        # Filled slots should have all-zero fields
        for event_idx in range(len(arr)):
            original_count = len(arr[event_idx])
            for slot_idx in range(original_count, 4):
                filled = result[event_idx][slot_idx]
                assert float(filled["pt"]) == 0.0
                assert float(filled["eta"]) == 0.0


# ---------------------------------------------------------------------------
# pad_and_fill — axis and negative axis
# ---------------------------------------------------------------------------


class TestPadAndFillAxis:
    """pad_and_fill with different axis values."""

    def test_negative_axis(self) -> None:
        arr = ak.Array([[[1, 2], [3, 4, 5]], [[6]]])
        result = pad_and_fill(arr, 3, value=-1, axis=-1)
        assert result.to_list() == [[[1, 2, -1], [3, 4, 5]], [[6, -1, -1]]]

    def test_axis_2(self) -> None:
        arr = ak.Array([[[1, 2], [3]], [[4, 5, 6]]])
        result = pad_and_fill(arr, 3, value=0, axis=2)
        assert result.to_list() == [[[1, 2, 0], [3, 0, 0]], [[4, 5, 6]]]


# ---------------------------------------------------------------------------
# pad_and_fill — NumpyArray with inner_shape (pure NumPy input)
# ---------------------------------------------------------------------------


class TestPadAndFillNumpyInnerShape:
    """Arrays from pure NumPy where all dims live in NumpyArray.inner_shape."""

    def test_numpy_3d_pad_axis_1(self) -> None:
        """Shape (N, M, K) from NumPy — pad axis=1 should fill with shape (K,)."""
        data = np.arange(30).reshape(3, 5, 2)
        # Make axis=1 ragged so padding is needed
        arr = ak.Array([data[0][:2], data[1][:4], data[2][:1]])
        result = pad_and_fill(arr, 5, value=-1)
        assert get_shape(result) == (3, 5, 2)
        # Filled slots should be [-1, -1]
        assert result[0][2].to_list() == [-1, -1]
        assert result[2][1].to_list() == [-1, -1]

    def test_numpy_backed_build_filler(self) -> None:
        """Pure NumPy ak.Array: filler at axis=1 should be shape (K,)."""

        ak_arr = ak.Array(np.random.rand(10, 150, 9))
        layout, skip = _layout_at_axis(ak_arr.layout, 1)
        filler = build_filler(layout, 0, skip)
        assert isinstance(filler, np.ndarray)
        assert filler.shape == (9,)
        assert filler.dtype == np.float64

    def test_numpy_4d_pad_axis_1(self) -> None:
        """Shape (N, A, B, C) from NumPy — filler at axis=1 is shape (B, C)."""

        ak_arr = ak.Array(np.zeros((5, 3, 4, 2)))
        layout, skip = _layout_at_axis(ak_arr.layout, 1)
        filler = build_filler(layout, -1, skip)
        assert isinstance(filler, np.ndarray)
        assert filler.shape == (4, 2)

    def test_numpy_4d_pad_axis_2(self) -> None:
        """Shape (N, A, B, C) from NumPy — filler at axis=2 is shape (C,)."""

        ak_arr = ak.Array(np.zeros((5, 3, 4, 2)))
        layout, skip = _layout_at_axis(ak_arr.layout, 2)
        filler = build_filler(layout, -1, skip)
        assert isinstance(filler, np.ndarray)
        assert filler.shape == (2,)