import numpy as np

from generic.get_prototypes_generic import (split_development_questions,
                                             split_development_questions_stratified)
from generic.utils_generic import format_winogrande_eval_prompt


def test_winogrande_paper_split_is_800_200_and_reproducible():
    q_indices = np.repeat(np.arange(1000), 2)
    fit_a, validation_a = split_development_questions(q_indices, 0.2, 42)
    fit_b, validation_b = split_development_questions(q_indices, 0.2, 42)
    assert len(fit_a) == 800
    assert len(validation_a) == 200
    assert not np.intersect1d(fit_a, validation_a).size
    np.testing.assert_array_equal(fit_a, fit_b)
    np.testing.assert_array_equal(validation_a, validation_b)


def test_winogrande_matched_prompt_is_shared_shape():
    prompt = format_winogrande_eval_prompt("The _ ran.", "dog", "tree")
    assert prompt.endswith("A:")
    assert "1) dog" in prompt and "2) tree" in prompt


def test_mmlu_split_is_400_100_per_category():
    q_indices, categories = [], []
    for category in range(4):
        for question in range(category * 500, (category + 1) * 500):
            q_indices.extend([question] * 4)
            categories.extend([category] * 4)
    fit, validation = split_development_questions_stratified(
        q_indices, categories, validation_fraction=0.2, seed=42)
    assert len(fit) == 1600
    assert len(validation) == 400
    assert set(fit).isdisjoint(set(validation))
    for category in range(4):
        lower, upper = category * 500, (category + 1) * 500
        assert sum(lower <= value < upper for value in fit) == 400
        assert sum(lower <= value < upper for value in validation) == 100


def test_copa_development_split_is_320_80_and_reproducible():
    q_indices = np.repeat(np.arange(400), 2)
    fit_a, validation_a = split_development_questions(q_indices, 0.2, 42)
    fit_b, validation_b = split_development_questions(q_indices, 0.2, 42)
    assert len(fit_a) == 320
    assert len(validation_a) == 80
    assert set(fit_a).isdisjoint(set(validation_a))
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(validation_a, validation_b)


def test_boolq_development_split_is_800_200():
    q_indices = np.repeat(np.arange(1000), 2)
    fit, validation = split_development_questions(q_indices, 0.2, 42)
    assert len(fit) == 800
    assert len(validation) == 200
    assert set(fit).isdisjoint(set(validation))
