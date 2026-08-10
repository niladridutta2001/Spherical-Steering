import numpy as np

from generic.get_prototypes_generic import split_development_questions
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
