import pytest
import torch

from ellipsoid_steering.activation_collector import ActivationCollector


def test_minimal_huggingface_transformer_collection():
    transformers=pytest.importorskip('transformers')
    config=transformers.GPT2Config(n_layer=2,n_head=2,n_embd=16,n_positions=16,vocab_size=31)
    model=transformers.GPT2Model(config).eval()
    result=ActivationCollector(model,[1]).collect({'input_ids':torch.randint(0,31,(1,5))})
    assert result[1].shape==(1,5,16)
