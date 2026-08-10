import numpy as np
import torch

from ellipsoid_geometry import (class_statistics, effective_weight_denominator,
    fit_ellipsoid_geometry, whiten_numpy, color_numpy, whiten_torch, color_torch)
from ellipsoidal_steering import ellipsoidal_geometric_logic, ellipsoidal_baukit_hook_fn
from truthfulqa_prompts import scoring_positions
from get_prototypes import load_sample_weights
from steering_artifacts import load_steering_artifact


def test_center_modes_with_imbalanced_weights():
    X=np.array([[0.,0.],[2.,0.],[10.,4.]])
    y=np.array([0,0,1]); w=np.array([1.,1.,1.])
    mT,mH,zero=class_statistics(X,y,w,"zero")
    _,_,global_center=class_statistics(X,y,w,"global")
    _,_,mid=class_statistics(X,y,w,"class-midpoint")
    np.testing.assert_array_equal(zero,np.zeros(2))
    np.testing.assert_allclose(global_center,np.average(X,axis=0,weights=w))
    np.testing.assert_allclose(mid,.5*(mT+mH))
    assert not np.allclose(global_center,mid)


def test_scoring_position_boundaries():
    P,L=11,4; positions=list(scoring_positions(P,L))
    assert positions[0]==P-1 and positions[-1]==P+L-2 and len(positions)==L
    assert min(positions)>=P-1 and P+L-1 not in positions


def test_answer_weights_and_weighted_diagonal_covariance():
    lengths=[2,3]
    weights=np.concatenate([np.full(L,1/L) for L in lengths])
    assert np.isclose(weights[:2].sum(),1) and np.isclose(weights[2:].sum(),1)
    X=np.array([[0.],[2.],[4.],[6.],[8.]])
    y=np.array([0,0,1,1,1])
    fit=fit_ellipsoid_geometry(X,y,"ellipsoid-diag",shrinkage=0,
        variance_floor=1e-12,sample_weights=weights,center_mode="global")
    center=np.sum(weights[:,None]*X,axis=0)/weights.sum()
    residual=np.where((y==1)[:,None],X-np.average(X[y==1],axis=0,weights=weights[y==1]),
                      X-np.average(X[y==0],axis=0,weights=weights[y==0]))
    expected=np.sum(weights[:,None]*residual**2,axis=0)/effective_weight_denominator(weights)
    np.testing.assert_allclose(fit["center"],center)
    np.testing.assert_allclose(fit["diag_var"],expected)


def _geometries(p):
    diag={"diag_var":torch.tensor([.5,1.,2.,3.]),"whitening_power":p}
    q,_=torch.linalg.qr(torch.randn(4,2))
    low={"basis":q,"eigvals":torch.tensor([.4,2.]),"residual_var":torch.tensor(1.2),
         "whitening_power":p}
    return diag,low


def test_power_zero_identity_and_roundtrip():
    x=torch.randn(5,4)
    for geometry in _geometries(0):
        torch.testing.assert_close(whiten_torch(x,geometry),x)
        torch.testing.assert_close(color_torch(x,geometry),x)
    for p in (.125,.25,.375,.5):
        for geometry in _geometries(p):
            torch.testing.assert_close(color_torch(whiten_torch(x,geometry),geometry),x,
                                       atol=3e-6,rtol=3e-6)


def test_powered_lowrank_matches_explicit_matrix():
    torch.manual_seed(5); x=torch.randn(3,4)
    for p in (0,.25,.5):
        geometry=_geometries(p)[1]; U=geometry["basis"]; lam=geometry["eigvals"]
        residual=geometry["residual_var"]
        covariance=U@torch.diag(lam)@U.T+residual*(torch.eye(4)-U@U.T)
        tau=torch.trace(covariance)/4
        eig,vec=torch.linalg.eigh(covariance)
        W=vec@torch.diag(torch.pow(eig/tau,-p))@vec.T
        torch.testing.assert_close(whiten_torch(x,geometry),x@W,atol=2e-5,rtol=2e-5)


def test_metric_radius_preserved_for_all_powers():
    torch.manual_seed(6); center=torch.randn(4); mu=torch.nn.functional.normalize(torch.randn(4),dim=0)
    for p in (0,.125,.25,.375,.5):
        for geometry in _geometries(p):
            h=center+torch.randn(4)
            before=whiten_torch(h-center,geometry).norm()
            out,_=ellipsoidal_geometric_logic(h,mu,-mu,center,geometry,20,.7,-1)
            after=whiten_torch(out-center,geometry).norm()
            torch.testing.assert_close(after,before,atol=1e-5,rtol=1e-5)


def test_bounded_hook_and_generation_compatibility():
    mu=torch.tensor([1.,0.,0.]); geom={"diag_var":torch.ones(3),"whitening_power":.5}
    h=torch.tensor([[[-1.,0,0],[-1.,0,0],[-1.,0,0],[-1.,0,0]]])
    kw=dict(layer_name="x",mu_T=mu,mu_H=-mu,center=torch.zeros(3),geometry=geom,
            kappa=20.,alpha=.5,beta=-1.)
    bounded=ellipsoidal_baukit_hook_fn(h.clone(),start_idx=1,end_idx_exclusive=3,**kw)
    assert torch.equal(bounded[:,0],h[:,0]) and torch.equal(bounded[:,3],h[:,3])
    assert not torch.equal(bounded[:,1],h[:,1])
    unbounded=ellipsoidal_baukit_hook_fn(h.clone(),start_idx=1,**kw)
    assert not torch.equal(unbounded[:,3],h[:,3])


def test_numpy_powered_roundtrip():
    geometry={"diag_var":np.array([.2,1.,5.]),"whitening_power":.375}
    x=np.random.default_rng(2).normal(size=(4,3))
    np.testing.assert_allclose(color_numpy(whiten_numpy(x,geometry),geometry),x,rtol=1e-12,atol=1e-12)


def test_legacy_features_use_uniform_weights():
    np.testing.assert_array_equal(load_sample_weights({}, 4), np.ones(4))


def test_old_ellipsoid_artifact_defaults(tmp_path):
    path=tmp_path/"old_ellipsoid.npz"
    np.savez(path, geometry=np.array("ellipsoid-diag"),
             mu_T=np.array([1.,0.]), mu_H=np.array([-1.,0.]),
             center=np.zeros(2), diag_var=np.array([1.,2.]))
    artifact=load_steering_artifact(path)
    assert artifact["center_mode"]=="global"
    assert artifact["whitening_power"]==.5
