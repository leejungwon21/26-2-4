"""Recreate missing model artifacts from SAVED selections, without tuning/test evaluation.

Use after cloning a completed experiment. Does not rewrite committed results or selections.
"""
import json
import pathlib
import sys

import joblib
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/model'))
from train import (config, make_split, tables, load_table, design, feature_names,
                   preprocessor, sha, save_json)


def main():
    make_split()
    execution = config('training'); output = ROOT/execution['output']; artifacts = ROOT/execution['artifacts']
    artifacts.mkdir(parents=True, exist_ok=True)
    record = []
    for key, path in tables():
        d = load_table(key, path); train = d[d['split'].eq('train')]
        for model_id in ['ridge', 'random_forest', 'catboost']:
            selected = json.loads((output/'selections'/f'{key}_{model_id}_v1.json').read_text(encoding='utf-8'))
            dest = artifacts/f'{key}_{model_id}.joblib'
            if dest.exists():
                if sha(dest) != selected['artifact_sha256']:
                    raise ValueError(f'Existing artifact mismatch: {dest.name}')
                continue
            params = selected['params']
            if model_id == 'catboost':
                model = CatBoostRegressor(**params, loss_function='RMSE', random_seed=config('models')['seed'],
                    thread_count=execution['catboost_threads'], verbose=False, allow_writing_files=False)
                model.fit(Pool(design(train), train.y, cat_features=feature_names()[1], weight=train.weight))
            else:
                estimator = Ridge(**params) if model_id == 'ridge' else RandomForestRegressor(
                    **params, random_state=config('models')['seed'], n_jobs=execution['forest_threads'])
                model = Pipeline([('preprocess', preprocessor(model_id == 'ridge')), ('model', estimator)])
                with threadpool_limits(limits=1):
                    model.fit(design(train), train.y, model__sample_weight=train.weight)
            joblib.dump(model, dest, compress=3)
            record.append({'table': key, 'model': model_id, 'sha256': sha(dest),
                'same_serialized_bytes_as_original': sha(dest) == selected['artifact_sha256'],
                'test_evaluated': False, 'parameters_retuned': False})
            save_json(record, artifacts/'refit_manifest.json')
            print(key, model_id, 'refit from saved selection', flush=True)


if __name__ == '__main__':
    main()
