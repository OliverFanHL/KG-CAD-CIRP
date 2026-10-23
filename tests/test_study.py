"""Invariants of final offline evaluation, independent of the packaged corpus."""
from itertools import product
import math
import numpy as np
import pytest
from cadkg.study.metrics import grade_limits, prepare_limits, rank_bounds
from cadkg.study.ranking import archive_order, rerank, score_only


def test_hybrid_reranking_preserves_eligibility_tail_and_ties():
    gallery = ['a', 'b', 'c', 'd']
    order = np.asarray([2, 1, 0, 3])
    scores = np.asarray([.5, .5, .5, .9])
    ranked, transformed = rerank(order, scores, ['c', 'a'], {}, gallery)
    assert ranked.tolist() == [0, 2, 1, 3]
    assert set(ranked) == set(order)
    assert transformed[:2].tolist() == [.5, .5]
    assert transformed[2:].tolist() == [.5, .9]
    assert score_only(order, ['c', 'a'], {}, gallery).tolist() == [0, 2, 1, 3]


def test_partial_judgments_bound_all_admissible_ndcg_completions():
    lower, upper = grade_limits([-1, 2, 1, 2], ['pass', 'unknown', 'pass', 'fail'])
    assert lower.tolist() == [0, 1, 1, 0]
    assert upper.tolist() == [2, 1, 1, 0]
    order = np.asarray([1, 0, 3, 2])
    bounds = rank_bounds(order, prepare_limits(lower, upper, (5,)))
    for completion in product(*[range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper)]):
        def dcg(grades):
            return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))
        exact = dcg([completion[i] for i in order]) / dcg(sorted(completion, reverse=True))
        assert bounds['ndcg@5'][0] <= exact <= bounds['ndcg@5'][1]


def test_saved_rankings_reject_duplicate_candidates():
    arrays = {'embedding__q__closed__kg_full__order': np.array([0, 0]),
              'embedding__q__closed__kg_full__score': np.array([.7, .6])}
    with pytest.raises(ValueError, match='ranking indices'):
        archive_order(arrays, 'embedding', 'q', 'kg_full', 2)
