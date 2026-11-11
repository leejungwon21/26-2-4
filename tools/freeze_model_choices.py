"""Select the model family by training OOF RMSE BEFORE opening test outcomes."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/model'))
from train import config, tables, make_split, load_table, baseline, save_csv, save_json, sha, MODEL_IDS
import pandas as pd


def main():
    make_split()
    output = ROOT/config('training')['output']
    if not (output/'training_complete_v1.json').exists():
        raise ValueError('Finish all training candidates first')
    if (output/'evaluation_started_v1.json').exists():
        raise ValueError('Family choice must be frozen before test evaluation')
    if (output/'cv_winners_v1.csv').exists():
        raise ValueError('Model-family choices already frozen')
    scores, winners = [], []
    for key, path in tables():
        d = load_table(key, path)
        train = d[d['split'].eq('train')].reset_index(drop=True)
        p = train[['cell_id', 'complex_id', 'dong_id', 'fold', 'y', 'weight']].copy()
        p['b0'] = 0.
        p['b1'] = np.nan
        for fold in sorted(train.fold.unique()):
            fit = train[train.fold.ne(fold)]; val = train[train.fold.eq(fold)]
            p.loc[val.index, 'b1'] = baseline(fit, val)
        assert p.b1.notna().all()
        save_csv(p, output/'oof'/f'{key}_baselines_v1.csv.xz')
        table_scores = {}
        for m in MODEL_IDS:
            if m in ['b0', 'b1']:
                score = float(np.sqrt(np.average((p[m]-p.y)**2, weights=p.weight)))
            else:
                s = json.loads((output/'selections'/f'{key}_{m}_v1.json').read_text(encoding='utf-8'))
                score = s['cv_rmse']
            table_scores[m] = score
            scores.append({'table': key, 'model': m, 'cv_rmse': score, 'train_cells': len(train)})
        winner = min(MODEL_IDS, key=lambda m: table_scores[m])
        winners.append({'table': key, 'selected_model': winner, 'cv_rmse': table_scores[winner],
                        'selection_uses_test': False})
    save_csv(pd.DataFrame(scores), output/'cv_comparison_v1.csv')
    save_csv(pd.DataFrame(winners), output/'cv_winners_v1.csv')
    save_json({'test_evaluated': False, 'family_choice_metric': 'pooled_weighted_cv_rmse',
        'tie_break': MODEL_IDS, 'script_sha256': sha(pathlib.Path(__file__)),
        'choices_sha256': sha(output/'cv_winners_v1.csv')}, output/'family_choice_manifest_v1.json')
    print(pd.DataFrame(winners).to_string(index=False))


if __name__ == '__main__':
    main()
