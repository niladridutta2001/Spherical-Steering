from sweep_ellipsoid import stable_config_hash, stage_configs, choose_best


def test_stable_hash_and_stage_dependencies():
    a={"rank":128,"center":"global","p":.5}
    b={"p":.5,"center":"global","rank":128}
    assert stable_config_hash(a)==stable_config_hash(b)
    power=stage_configs("power","class-midpoint")
    assert all(x["center_mode"]=="class-midpoint" for x in power)
    covariance=stage_configs("covariance","zero",.25)
    assert all(x["center_mode"]=="zero" and x["whitening_power"]==.25 for x in covariance)


def test_selection_and_resume_basis():
    rows=[{"stage":"center","status":"complete","MC_avg":.4,"selection_score":.6,"center_mode":"zero"},
          {"stage":"center","status":"complete","MC_avg":.5,"selection_score":.5,"center_mode":"global"},
          {"stage":"center","status":"failed","MC_avg":.9,"center_mode":"class-midpoint"}]
    assert choose_best(rows,"center")["center_mode"]=="zero"
