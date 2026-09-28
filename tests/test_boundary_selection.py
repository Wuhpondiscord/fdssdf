from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voynich_lab.boundary_selection import select_boundary_markov_order


def test_boundary_order_selection_returns_candidate_and_scores_all_orders():
    groups = {
        "A": "ab ab ab\nab ab",
        "B": "ab ab\nab ab ab",
        "C": "ab ab ab\nab ab ab",
    }
    result = select_boundary_markov_order(groups, candidates=(1, 2, 3), alpha=0.5)
    assert result.selected_order in {1, 2, 3}
    assert [row.order for row in result.scores] == [1, 2, 3]
    assert all(row.bits_per_symbol == row.bits_per_symbol for row in result.scores)
    assert all(row.held_out_symbols > 0 for row in result.scores)
    assert "held-out predictive log loss" in result.note


def test_boundary_order_selection_requires_multiple_groups():
    try:
        select_boundary_markov_order({"A": "abc def"})
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("expected a single group to be rejected")


def test_boundary_order_selection_rejects_nonpositive_alpha():
    try:
        select_boundary_markov_order({"A": "abc", "B": "abd"}, alpha=0)
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("expected alpha=0 to be rejected")
