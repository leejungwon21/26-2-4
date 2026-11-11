"""Frozen group split, train-only tuning and one-time held-out evaluation.

python src/model/train.py split
python src/model/train.py train
python src/model/train.py evaluate

Resume skips completed training artifacts only when input/config fingerprints match.
Evaluation requires all training selections and refuses to overwrite existing results.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import pathlib
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from catboost import CatBoostRegressor, Pool
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_IDS = ['b0', 'b1', 'ridge', 'random_forest', 'catboost']


@lru_cache(maxsize=None)
def config(name):
    return yaml.safe_load((ROOT / 'config' / f'{name}.yaml').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def save_csv(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n', float_format='%.12g')


def tables():
    for regime in config('regime')['regimes']:
        for method in config('training')['methods']:
            key = f'{regime["id"]}_{method}'
            yield key, ROOT / f'reports/final/cells/{key}_v1.csv.xz'


def input_fingerprint():
    paths = [p for _, p in tables()]
    paths += [ROOT / 'config' / f'{name}.yaml' for name in ['split', 'models', 'training', 'features', 'regime']]
    # Source fingerprint prevents stale caches after an implementation change.
    paths += [pathlib.Path(__file__)]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def split_dongs(dongs, share, seed):
    ordered = sorted(set(dongs))
    rng = np.random.default_rng(seed)
    test = set(rng.permutation(ordered)[:math.ceil(len(ordered)*share)])
    return pd.DataFrame({'dong_id': ordered, 'split': ['test' if d in test else 'train' for d in ordered]})


def make_split():
    settings = config('split'); output = ROOT / config('training')['output']
    manifest_path = output / 'split_manifest_v1.json'
    fingerprints = input_fingerprint()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['input_sha256'] != fingerprints:
            raise ValueError('Frozen split inputs/code changed; do not silently redefine the experiment')
        for name, expected in old['split_file_sha256'].items():
            if sha(output / name) != expected:
                raise ValueError('Frozen split assignments changed')
        print('Existing frozen split verified', flush=True)
        return
    frames = {key: pd.read_csv(path) for key, path in tables()}
    assignments = split_dongs(itertools.chain.from_iterable(d.dong_id for d in frames.values()), settings['test_share'], settings['seed'])
    split_map = assignments.set_index('dong_id')['split']
    records, summaries = [], []
    for key, d in frames.items():
        s = d[['cell_id', 'complex_id', 'dong_id']].copy()
        s['table'] = key; s['split'] = s.dong_id.map(split_map); s['fold'] = -1
        tr = np.flatnonzero(s['split'].eq('train'))
        folds = GroupKFold(settings['cv']['n_splits'])
        for fold, (_, va) in enumerate(folds.split(tr, groups=s.iloc[tr].dong_id)):
            s.loc[tr[va], 'fold'] = fold
        assert not set(s.loc[s['split'].eq('train'), 'dong_id']) & set(s.loc[s['split'].eq('test'), 'dong_id'])
        for fold in range(settings['cv']['n_splits']):
            va = s[s.fold.eq(fold)]; other = s[s['split'].eq('train') & s.fold.ne(fold)]
            assert not set(va.dong_id) & set(other.dong_id)
        records.append(s)
        for subset in ['train', 'test']:
            part = s[s['split'].eq(subset)]
            summaries.append({'table': key, 'split': subset, 'cells': len(part),
                'dongs': part.dong_id.nunique(), 'complexes': part.complex_id.nunique()})
    save_csv(assignments, output / 'dong_assignments_v1.csv')
    selected = assignments[assignments['split'].eq('test')].copy()
    selected[settings['group']] = selected.dong_id.str.split('|', n=1, expand=True)
    save_csv(selected, ROOT / settings['save_to'])
    save_csv(pd.concat(records, ignore_index=True), output / 'cell_splits_v1.csv.xz')
    save_csv(pd.DataFrame(summaries), output / 'split_summary_v1.csv')
    manifest = {'input_sha256': fingerprints, 'seed': settings['seed'], 'dongs': len(assignments),
        'test_dongs': len(selected), 'test_share_requested': settings['test_share'],
        'split_file_sha256': {p.name: sha(p) for p in [output/'dong_assignments_v1.csv', output/'cell_splits_v1.csv.xz']},
        'test_evaluated': False}
    save_json(manifest, manifest_path)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['input_sha256', 'split_file_sha256']}), flush=True)


def load_table(key, path):
    d = pd.read_csv(path).sort_values('cell_id').reset_index(drop=True)
    split = pd.read_csv(ROOT / config('training')['output'] / 'cell_splits_v1.csv.xz')
    split = split[split.table.eq(key)][['cell_id', 'split', 'fold']]
    return d.merge(split, on='cell_id', validate='one_to_one')


def feature_names():
    base = [s for group in config('features')['base'].values() for s in group]
    cats = [s['name'] for s in base if s.get('type') == 'category']
    names = [s['name'] for s in base]
    return names + [n+'_missing' for n in names], cats


def design(d):
    names, cats = feature_names()
    x = d[names].copy()
    for c in cats:
        x[c] = x[c].fillna(config('training')['category_missing']).astype(str)
    return x


def preprocessor(scale):
    names, cats = feature_names(); nums = [c for c in names if c not in cats]
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return ColumnTransformer([('numeric', Pipeline(steps), nums),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])


def baseline(d, x):
    means = d.groupby('자치구').apply(lambda g: np.average(g.y, weights=g.weight), include_groups=False)
    fallback = np.average(d.y, weights=d.weight)
    return x['자치구'].map(means).fillna(fallback).to_numpy()


def metric_values(y, prediction, w):
    y, prediction, w = map(lambda a: np.asarray(a, float), [y, prediction, w])
    residual = prediction-y
    ry, rp = rankdata(y), rankdata(prediction)
    ry -= np.average(ry, weights=w); rp -= np.average(rp, weights=w)
    den = np.sqrt(np.sum(w*ry**2)*np.sum(w*rp**2))
    correlation = float(np.sum(w*ry*rp)/den) if den > 0 else np.nan
    # Ties at a cutoff contribute equally, avoiding arbitrary apartment-ID ordering.
    tail_n = len(y)*config('training')['tail_share']
    _, inverse, count = np.unique(prediction, return_inverse=True, return_counts=True)
    before = np.cumsum(count)-count
    low_fraction = np.clip((tail_n-before)/count, 0, 1)[inverse]
    after = len(y)-np.cumsum(count)
    high_fraction = np.clip((tail_n-after)/count, 0, 1)[inverse]
    gap = np.average(y, weights=w*high_fraction)-np.average(y, weights=w*low_fraction)
    return {'RMSE': float(np.sqrt(np.average(residual**2, weights=w))),
            'MAE': float(np.average(abs(residual), weights=w)), 'Spearman': correlation,
            'top_bottom_20pct_gap': float(gap)}


def fit_candidate(model_id, params, train, folds, workers_cfg):
    x, y, w = design(train), train.y.to_numpy(), train.weight.to_numpy()
    oof = np.empty(len(train)); iterations = []; fold_stats = []
    for fold in sorted(set(folds)):
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if model_id == 'catboost':
            mcfg = config('models')['models']['catboost']
            model = CatBoostRegressor(**params, iterations=mcfg['iterations'],
                random_seed=config('models')['seed'], thread_count=workers_cfg['catboost_threads'],
                loss_function='RMSE', verbose=False, allow_writing_files=False)
            cats = feature_names()[1]
            model.fit(Pool(x.iloc[tr], y[tr], cat_features=cats, weight=w[tr]),
                eval_set=Pool(x.iloc[va], y[va], cat_features=cats, weight=w[va]),
                early_stopping_rounds=mcfg['early_stopping_rounds'], use_best_model=True)
            prediction = model.predict(x.iloc[va]); iterations.append(int(model.tree_count_))
        else:
            prep = preprocessor(model_id == 'ridge')
            xt, xv = prep.fit_transform(x.iloc[tr]), prep.transform(x.iloc[va])
            model = Ridge(**params) if model_id == 'ridge' else RandomForestRegressor(
                **params, random_state=config('models')['seed'], n_jobs=workers_cfg['forest_threads'])
            model.fit(xt, y[tr], sample_weight=w[tr]); prediction = model.predict(xv)
        oof[va] = prediction
        fold_stats.append(float(np.sqrt(np.average((prediction-y[va])**2, weights=w[va]))))
    score = float(np.sqrt(np.average((oof-y)**2, weights=w)))
    return {'params': params, 'cv_rmse': score, 'fold_rmse': fold_stats,
            'best_iterations': iterations}, oof


def train_all():
    make_split()
    execution = config('training'); output = ROOT / execution['output']; artifacts = ROOT / execution['artifacts']
    artifacts.mkdir(parents=True, exist_ok=True)
    if (output / 'test_metrics_v1.csv').exists():
        raise ValueError('Test already evaluated; refuse retraining this experiment')
    manifest = json.loads((output/'split_manifest_v1.json').read_text(encoding='utf-8'))
    for key, path in tables():
        data = load_table(key, path)
        train = data[data['split'].eq('train')].reset_index(drop=True)
        folds = train.fold.to_numpy()
        for model_id in ['ridge', 'random_forest', 'catboost']:
            selection_path = output / 'selections' / f'{key}_{model_id}_v1.json'
            model_path = artifacts / f'{key}_{model_id}.joblib'
            if selection_path.exists():
                saved = json.loads(selection_path.read_text(encoding='utf-8'))
                if not model_path.exists() or sha(model_path) != saved['artifact_sha256']:
                    raise ValueError('Missing/mismatched completed model artifact')
                print('Resume', key, model_id, flush=True); continue
            grid = list(ParameterGrid(config('models')['models'][model_id]['grid']))
            results = []
            started = time.monotonic()
            print('Tuning', key, model_id, len(grid), 'candidates', flush=True)
            with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=execution['workers']) as pool:
                pending = {pool.submit(fit_candidate, model_id, params, train, folds, execution): j for j, params in enumerate(grid)}
                for task in as_completed(pending):
                    result, oof = task.result(); result['candidate'] = pending[task]
                    results.append((result, oof))
                    print('CV', key, model_id, len(results), '/', len(grid), 'elapsed_s', round(time.monotonic()-started), flush=True)
            results.sort(key=lambda pair: (pair[0]['cv_rmse'], pair[0]['candidate']))
            selected, oof = results[0]
            params = selected['params'].copy()
            if model_id == 'catboost':
                params['iterations'] = max(1, int(np.median(selected['best_iterations'])))
                model = CatBoostRegressor(**params, loss_function='RMSE',
                    random_seed=config('models')['seed'], thread_count=execution['catboost_threads'],
                    verbose=False, allow_writing_files=False)
                model.fit(Pool(design(train), train.y, cat_features=feature_names()[1], weight=train.weight))
            else:
                estimator = Ridge(**params) if model_id == 'ridge' else RandomForestRegressor(
                    **params, random_state=config('models')['seed'], n_jobs=execution['forest_threads'])
                model = Pipeline([('preprocess', preprocessor(model_id == 'ridge')), ('model', estimator)])
                with threadpool_limits(limits=1):
                    model.fit(design(train), train.y, model__sample_weight=train.weight)
            joblib.dump(model, model_path, compress=3)
            cv_records = [r for r, _ in sorted(results, key=lambda pair: pair[0]['candidate'])]
            save_json(cv_records, output/'cv'/f'{key}_{model_id}_v1.json')
            pred = train[['cell_id', 'complex_id', 'dong_id', 'fold', 'y', 'weight']].copy()
            pred['prediction'] = oof
            save_csv(pred, output/'oof'/f'{key}_{model_id}_v1.csv.xz')
            save_json({'table': key, 'model': model_id, 'params': params, 'cv_rmse': selected['cv_rmse'],
                'train_cells': len(train), 'artifact_sha256': sha(model_path), 'input_sha256': manifest['input_sha256'],
                'elapsed_seconds': round(time.monotonic()-started, 2)}, selection_path)
            print('Selected', key, model_id, params, 'CV_RMSE', round(selected['cv_rmse'], 4), flush=True)
    save_json({'complete': True, 'tables': len(list(tables())), 'learned_models': 3*len(list(tables())),
        'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__,
        'test_evaluated': False}, output/'training_complete_v1.json')


def evaluate():
    execution = config('training'); output = ROOT/execution['output']; artifacts = ROOT/execution['artifacts']
    make_split()
    if not (output/'training_complete_v1.json').exists():
        raise ValueError('Freeze all model selections before evaluating test')
    destination = output/'test_metrics_v1.csv'
    if destination.exists() or (output/'evaluation_started_v1.json').exists():
        raise ValueError('Evaluation already started/completed. Use saved predictions; do not reopen test for tuning.')
    # Validate every model artifact before marking the test as opened.
    for key, _ in tables():
        for model_id in MODEL_IDS[2:]:
            s = json.loads((output/'selections'/f'{key}_{model_id}_v1.json').read_text(encoding='utf-8'))
            if sha(artifacts/f'{key}_{model_id}.joblib') != s['artifact_sha256']:
                raise ValueError('Model artifact changed')
    save_json({'started': True, 'tuning_frozen': True}, output/'evaluation_started_v1.json')
    metrics, intervals = [], []
    for key, path in tables():
        d = load_table(key, path); train = d[d['split'].eq('train')]; test = d[d['split'].eq('test')].reset_index(drop=True)
        predictions = {'b0': np.zeros(len(test)), 'b1': baseline(train, test)}
        for model_id in MODEL_IDS[2:]:
            model = joblib.load(artifacts/f'{key}_{model_id}.joblib')
            predictions[model_id] = model.predict(design(test))
        p = test[['cell_id', 'complex_id', 'dong_id', 'y', 'weight']].copy()
        for model_id, prediction in predictions.items():
            p[model_id] = prediction
        save_csv(p, output/'predictions'/f'{key}_test_v1.csv.xz')
        point = {m: metric_values(test.y, pred, test.weight) for m, pred in predictions.items()}
        groups = [np.asarray(v) for v in test.groupby('complex_id', sort=True).indices.values()]
        rng = np.random.default_rng(config('models')['seed'])
        bootstrap = {m: {metric: [] for metric in point[m]} for m in MODEL_IDS}
        deltas = {m: [] for m in MODEL_IDS}
        uncertainty = config('models')['uncertainty']
        for _ in range(uncertainty['n']):
            take = np.concatenate([groups[j] for j in rng.integers(0, len(groups), size=len(groups))])
            scores = {m: metric_values(test.y.to_numpy()[take], pred[take], test.weight.to_numpy()[take]) for m, pred in predictions.items()}
            for m in MODEL_IDS:
                for metric, value in scores[m].items():
                    bootstrap[m][metric].append(value)
                deltas[m].append(scores[m]['RMSE']-scores['b1']['RMSE'])
        quantiles = [(1-uncertainty['level'])/2, 1-(1-uncertainty['level'])/2]
        for m in MODEL_IDS:
            metrics.append({'table': key, 'model': m, 'test_cells': len(test), 'test_complexes': len(groups),
                **point[m], 'rmse_minus_b1': point[m]['RMSE']-point['b1']['RMSE'],
                'eligible_for_interpretation': m in MODEL_IDS[2:] and point[m]['RMSE'] < point['b1']['RMSE']})
            for metric, values in {**bootstrap[m], 'rmse_minus_b1': deltas[m]}.items():
                finite = np.asarray(values)[np.isfinite(values)]
                low, high = np.quantile(finite, quantiles) if len(finite) else (np.nan, np.nan)
                intervals.append({'table': key, 'model': m, 'metric': metric, 'low': low, 'high': high,
                                  'valid_replicates': len(finite), 'requested_replicates': uncertainty['n']})
        print('Evaluated', key, 'test cells', len(test), flush=True)
    save_csv(pd.DataFrame(metrics), destination)
    save_csv(pd.DataFrame(intervals), output/'test_intervals_v1.csv')
    save_json({'complete': True, 'models_evaluated': len(metrics), 'bootstrap_replicates': config('models')['uncertainty']['n'],
        'bootstrap_unit': 'complex_id', 'bootstrap_conditional_on_fixed_models': True}, output/'evaluation_complete_v1.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['split', 'train', 'evaluate'])
    args = parser.parse_args()
    {'split': make_split, 'train': train_all, 'evaluate': evaluate}[args.phase]()
