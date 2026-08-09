import numpy as np
import pytest

from get_prototypes import split_question_folds


def test_paper_protocol_splits_are_disjoint_complete_and_reproducible():
    q_indices = np.repeat(np.arange(817), 2)
    first = list(split_question_folds(q_indices, validation_fraction=0.2,
                                      shuffle_folds=True, seed=42))
    second = list(split_question_folds(q_indices, validation_fraction=0.2,
                                       shuffle_folds=True, seed=42))
    assert len(first) == 2
    for a, b in zip(first, second):
        fold, train, validation, test = a
        assert fold == b[0]
        np.testing.assert_array_equal(train, b[1])
        np.testing.assert_array_equal(validation, b[2])
        np.testing.assert_array_equal(test, b[3])
        assert not set(train) & set(validation)
        assert not set(train) & set(test)
        assert not set(validation) & set(test)
        assert set(train) | set(validation) | set(test) == set(range(817))
        assert len(validation) in (81, 82)


def test_legacy_split_has_no_validation_questions():
    fold = next(split_question_folds(np.arange(20)))
    assert len(fold[2]) == 0


@pytest.mark.parametrize("value", [-0.1, 1.0])
def test_invalid_validation_fraction(value):
    with pytest.raises(ValueError):
        list(split_question_folds(np.arange(20), validation_fraction=value))
