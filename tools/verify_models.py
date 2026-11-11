"""Verify frozen group isolation and saved held-out predictions without refitting."""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/model'))
from train import config, feature_names, input_fingerprint, tables, save_json


def main():
    output = ROOT/config('training')['output']
    manifest = json.loads((output/'split_manifest_v1.json').read_text(encoding='utf-8'))
    assert manifest['input_sha256'] == input_fingerprint()
    for filename, expected in manifest['split_file_sha256'].items():
        assert hashlib.sha256((output/filename).read_bytes()).hexdigest() == expected
    assignment = pd.read_csv(output/'dong_assignments_v1.csv').set_index('dong_id')['split']
    splits = pd.read_csv(output/'cell_splits_v1.csv.xz')
    total = 0
    oof_checked = 0
    for key, path in tables():
        source = pd.read_csv(path).set_index('cell_id')
        s = splits[splits.table.eq(key)].set_index('cell_id')
        assert set(s.index) == set(source.index)
        assert not s.index.duplicated().any()
        assert s['split'].eq(s.dong_id.map(assignment)).all()
        tr, te = s[s['split'].eq('train')], s[s['split'].eq('test')]
        assert not set(tr.dong_id) & set(te.dong_id)
        assert not set(tr.complex_id) & set(te.complex_id)
        assert te.fold.eq(-1).all()
        for fold in range(config('split')['cv']['n_splits']):
            assert not set(tr[tr.fold.eq(fold)].dong_id) & set(tr[tr.fold.ne(fold)].dong_id)
        for model_id in ['ridge', 'random_forest', 'catboost']:
            selected_path = output/'selections'/f'{key}_{model_id}_v1.json'
            if not selected_path.exists():
                continue
            selected = json.loads(selected_path.read_text(encoding='utf-8'))
            oof = pd.read_csv(output/'oof'/f'{key}_{model_id}_v1.csv.xz').set_index('cell_id')
            assert set(oof.index) == set(tr.index)
            assert not oof.index.duplicated().any()
            assert oof.fold.eq(tr.loc[oof.index, 'fold']).all()
            np.testing.assert_allclose(oof.y, source.loc[oof.index, 'y'], atol=1e-8, rtol=1e-10)
            np.testing.assert_allclose(oof.weight, source.loc[oof.index, 'weight'])
            expected = np.sqrt(np.average((oof.prediction-oof.y)**2, weights=oof.weight))
            assert np.isclose(expected, selected['cv_rmse'], atol=1e-8, rtol=1e-9)
            candidates = json.loads((output/'cv'/f'{key}_{model_id}_v1.json').read_text(encoding='utf-8'))
            best = min(candidates, key=lambda row: (row['cv_rmse'], row['candidate']))
            assert best['cv_rmse'] == selected['cv_rmse']
            params = selected['params'].copy()
            if model_id == 'catboost':
                assert params.pop('iterations') == max(1, int(np.median(best['best_iterations'])))
            assert params == best['params']
            oof_checked += 1
        total += len(s)
    result = {'group_isolation_passed': True, 'assigned_rows': total,
              'dongs': len(assignment), 'test_dongs': int(assignment.eq('test').sum()),
              'explicit_feature_count_including_missing_indicators': len(feature_names()[0]),
              'oof_models_checked': oof_checked}
    if (output/'cv_winners_v1.csv').exists():
        comparison = pd.read_csv(output/'cv_comparison_v1.csv')
        winners = pd.read_csv(output/'cv_winners_v1.csv')
        family_manifest = json.loads((output/'family_choice_manifest_v1.json').read_text(encoding='utf-8'))
        assert hashlib.sha256((output/'cv_winners_v1.csv').read_bytes()).hexdigest() == family_manifest['choices_sha256']
        for row in winners.itertuples():
            sub = comparison[comparison.table.eq(row.table)]
            best = sub.sort_values('cv_rmse', kind='stable').iloc[0]
            assert row.selected_model == best.model
            assert np.isclose(row.cv_rmse, best.cv_rmse, atol=1e-8, rtol=1e-9)
            assert not row.selection_uses_test
            p = pd.read_csv(output/'oof'/f'{row.table}_baselines_v1.csv.xz')
            expected_ids = splits[splits.table.eq(row.table) & splits['split'].eq('train')].cell_id
            assert set(p.cell_id) == set(expected_ids)
            for model_id in ['b0', 'b1']:
                value = np.sqrt(np.average((p[model_id]-p.y)**2, weights=p.weight))
                saved = sub[sub.model.eq(model_id)].cv_rmse.iloc[0]
                assert np.isclose(value, saved, atol=1e-8, rtol=1e-9)
        result['cv_model_families_checked'] = len(winners)
    metrics_path = output/'test_metrics_v1.csv'
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        checked = 0
        for key, path in tables():
            source = pd.read_csv(path).set_index('cell_id')
            p = pd.read_csv(output/'predictions'/f'{key}_test_v1.csv.xz').set_index('cell_id')
            s = splits[splits.table.eq(key) & splits['split'].eq('test')]
            assert set(p.index) == set(s.cell_id)
            np.testing.assert_allclose(p.y, source.loc[p.index, 'y'], atol=1e-8, rtol=1e-10)
            np.testing.assert_allclose(p.weight, source.loc[p.index, 'weight'])
            assert np.isfinite(p[['y', 'weight', 'b0', 'b1', 'ridge', 'random_forest', 'catboost']]).all().all()
            for row in metrics[metrics.table.eq(key)].itertuples():
                error = p[row.model]-p.y
                assert np.isclose(row.RMSE, np.sqrt(np.sum(p.weight*error**2)/p.weight.sum()), atol=1e-8)
                assert np.isclose(row.MAE, np.sum(p.weight*abs(error))/p.weight.sum(), atol=1e-8)
                if row.model == 'b0':
                    assert p.b0.eq(0).all() and np.isnan(row.Spearman)
                    assert abs(row.top_bottom_20pct_gap) < 1e-8
                checked += 1
        ci = pd.read_csv(output/'test_intervals_v1.csv')
        assert len(ci) == len(metrics)*5
        assert (ci.low.dropna() <= ci.high.dropna()).all()
        defined = ci[ci.metric.ne('Spearman')]
        assert defined.valid_replicates.eq(config('models')['uncertainty']['n']).all()
        self_delta = ci[ci.model.eq('b1') & ci.metric.eq('rmse_minus_b1')]
        assert self_delta.low.eq(0).all() and self_delta.high.eq(0).all()
        result.update(test_predictions_verified=True, model_scores_checked=checked,
                      intervals_checked=len(ci), model_refit_performed=False)
    save_json(result, output/'verification_v1.json')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
