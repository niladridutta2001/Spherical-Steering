from generic.sweep_ellipsoid_generic import configurations, config_hash, best_row


def test_wino_geometry_sweep_counts_and_dependencies():
    assert len(configurations('center')) == 3
    powers = configurations('power', 'class-midpoint')
    assert len(powers) == 5
    assert all(x['center_mode'] == 'class-midpoint' for x in powers)
    covariance = configurations('covariance', 'zero', 0.25)
    assert len(covariance) == 16
    assert all(x['center_mode'] == 'zero' and x['whitening_power'] == 0.25
               for x in covariance)


def test_wino_sweep_hash_and_accuracy_selection():
    assert config_hash({'a': 1, 'b': 2}) == config_hash({'b': 2, 'a': 1})
    rows = [
        {'stage': 'power', 'status': 'complete', 'accuracy': 0.6,
         'trigger_rate': 0.7, 'whitening_power': 0.25},
        {'stage': 'power', 'status': 'complete', 'accuracy': 0.61,
         'trigger_rate': 0.5, 'whitening_power': 0.5},
    ]
    assert best_row(rows, 'power')['whitening_power'] == 0.5
