import torch

from ellipsoid_steering.activation_collector import ActivationCollector
from ellipsoid_steering.hooks import register_layer_hooks
from ellipsoid_steering.config import EllipsoidSteeringConfig
from ellipsoid_steering.controller import EllipsoidSteeringController, SteeringContext
from ellipsoid_steering.covariance import LowRankCovariance
from ellipsoid_steering.hallucination_modes import HallucinationMode


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model=torch.nn.Module(); self.model.layers=torch.nn.ModuleList(
            [torch.nn.Linear(4,4,bias=False),torch.nn.Linear(4,4,bias=False)])
    def forward(self, input_ids):
        x=input_ids
        for layer in self.model.layers: x=layer(x)
        return x


def test_collector_detaches_only_target_layers():
    model=TinyModel(); values=torch.randn(2,3,4,requires_grad=True)
    result=ActivationCollector(model,[1]).collect({'input_ids':values})
    assert list(result)==[1] and result[1].shape==(2,3,4) and not result[1].requires_grad


def test_edit_hook_preserves_shape_and_does_not_mutate_input():
    model=TinyModel(); values=torch.randn(2,3,4); original=values.clone()
    with register_layer_hooks(model.model.layers,[0],lambda _i,h: h+1,edit=True):
        output=model(values)
    assert torch.equal(values,original) and output.shape==values.shape


def test_two_pass_controller_edits_tiny_model_without_gradients():
    model=TinyModel().eval(); q,_=torch.linalg.qr(torch.randn(4,2))
    covariance=LowRankCovariance(torch.zeros(4),q,torch.tensor([2.,1.]),torch.tensor(.5))
    mode=HallucinationMode(torch.zeros(4),q[:,:1],torch.tensor([2.]),10)
    config=EllipsoidSteeringConfig([1],routing_mode='teacher_forced',
        steering_threshold=0,uniform_beta=.2,beta_max=.5)
    controller=EllipsoidSteeringController(model,{1:covariance},{1:[mode]},config)
    original=torch.randn(1,3,4); masked=original*.5
    context=SteeringContext(controller.routing_pass({'input_ids':masked}))
    with controller.enabled(context): output=model(original)
    assert output.shape==original.shape and not output.requires_grad
    assert 1 in context.diagnostics
