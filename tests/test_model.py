"""Leakage and metric checks on synthetic data, never on the real test outcomes."""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('training', ROOT/'src/model/train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def synthetic(n=30):
    names, cats = training.feature_names()
    d = pd.DataFrame({name: np.arange(n, dtype=float) % 7 for name in names})
    for c in cats:
        d[c] = ['a', 'b', None] * (n//3)
    d['y'] = np.arange(n)*.3
    d['weight'] = 1 + np.arange(n) % 3
    d['fold'] = np.arange(n) % 5
    d['y_future_leak'] = d.y
    return d


def test_split_is_reproducible_order_independent_and_whole_dong():
    dongs = [f'구|동{i}' for i in range(11)]
    a = training.split_dongs(dongs, .2, 3)
    b = training.split_dongs(list(reversed(dongs))*3, .2, 3)
    pd.testing.assert_frame_equal(a, b)
    assert a['split'].eq('test').sum() == 3
    assert not a.dong_id.duplicated().any()


def test_design_is_explicit_allowlist_and_imputer_uses_only_train():
    d = synthetic()
    x = training.design(d)
    assert set(x) == set(training.feature_names()[0])
    assert not {'y', 'weight', 'fold', 'y_future_leak'} & set(x)
    x.loc[:2, '세대수'] = [1., np.nan, 3.]
    prep = training.preprocessor(True)
    prep.fit(x.iloc[:3])
    imputer = prep.named_transformers_['numeric'].named_steps['impute']
    names, cats = training.feature_names(); nums = [n for n in names if n not in cats]
    assert imputer.statistics_[nums.index('세대수')] == 2
    before = imputer.statistics_.copy()
    test = x.iloc[3:].copy(); test['세대수'] = 1e9; test['자치구'] = 'unseen'
    assert np.isfinite(prep.transform(test)).all()
    np.testing.assert_array_equal(before, imputer.statistics_)


def test_metrics_weighting_spearman_and_prediction_ties():
    y = np.array([1., 2., 5., 9.]); pred = np.array([2., 3., 4., 8.])
    result = training.metric_values(y, pred, np.ones(4))
    assert result['RMSE'] == 1
    assert result['MAE'] == 1
    assert result['Spearman'] == pytest.approx(spearmanr(y, pred).statistic)
    constant = training.metric_values(y, np.zeros(4), np.array([1, 2, 3, 4]))
    assert np.isnan(constant['Spearman'])
    assert constant['top_bottom_20pct_gap'] == pytest.approx(0)
    assert constant['RMSE'] == pytest.approx(np.sqrt(np.average(y*y, weights=[1, 2, 3, 4])))
    tied = np.array([1., 1., 4., 4.]); order = [3, 1, 0, 2]
    a = training.metric_values(y, tied, np.ones(4))
    b = training.metric_values(y[order], tied[order], np.ones(4))
    assert a == pytest.approx(b)


def test_baseline_unseen_district_uses_training_global_weighted_mean():
    d = pd.DataFrame({'자치구': ['a', 'a', 'b'], 'y': [1, 4, 10], 'weight': [1, 2, 1]})
    pred = training.baseline(d, pd.DataFrame({'자치구': ['a', 'b', 'new']}))
    np.testing.assert_allclose(pred, [3, 10, 4.75])


@pytest.mark.parametrize('model_id,params', [('ridge', {'alpha': 1}),
    ('random_forest', {'n_estimators': 3, 'max_depth': 2, 'min_samples_leaf': 2}),
    ('catboost', {'depth': 2, 'learning_rate': .1, 'l2_leaf_reg': 3})])
def test_cv_models_produce_full_oof_predictions(model_id, params, monkeypatch):
    original = training.config
    def small(name):
        if name == 'models':
            return {'seed': 42, 'models': {'catboost': {'iterations': 5, 'early_stopping_rounds': 2}}}
        return original(name)
    monkeypatch.setattr(training, 'config', small)
    d = synthetic()
    result, oof = training.fit_candidate(model_id, params, d, d.fold.to_numpy(), {'catboost_threads': 1, 'forest_threads': 1})
    assert len(oof) == len(d) and np.isfinite(oof).all()
    assert result['cv_rmse'] == pytest.approx(np.sqrt(np.average((oof-d.y)**2, weights=d.weight)))
    assert len(result['fold_rmse']) == 5
