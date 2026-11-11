"""Check saved cells against independent raw-transaction calculations.

python tools/verify_cells.py
Checks all table invariants and 20 deterministic cells per regime/method (240 targets).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/cells'))
from build_cells import bounds, month


def main():
    load = lambda n: yaml.safe_load((ROOT / 'config' / f'{n}.yaml').read_text(encoding='utf-8'))
    cfg, target, features = [load(n) for n in ['cells', 'target', 'features']]
    regimes = load('regime')['regimes']
    out = ROOT / cfg['output']
    raw = pd.read_csv(ROOT / cfg['source'], dtype=str, keep_default_na=False)
    n = len(raw)
    raw = raw[raw['위도'].ne('') & raw['경도'].ne('') & raw['연식_기준'].ne('준공 전 거래')].copy()
    excluded = pd.read_csv(out / 'excluded_trades_v1.csv', dtype=str)
    assert len(raw) + len(excluded) == n
    assert not set(raw['거래ID']) & set(excluded['거래ID'])
    raw['period'] = pd.to_datetime(raw['계약일']).dt.to_period('M').astype('int64')
    raw['area'] = raw['전용면적(㎡)'].astype(float)
    raw['area_round'] = np.round(raw.area/target['cell']['area_rounding_m2'])*target['cell']['area_rounding_m2']
    raw['price'] = raw['거래금액_만원'].str.replace(',', '').astype(float)
    idx = pd.read_csv(ROOT / 'data/regime/seoul_index_2017_2026.csv', index_col=0)[target['index_for_time_adjustment']]
    raw['index'] = pd.to_datetime(raw['계약일']).dt.strftime('%Y-%m').map(idx)
    groups = raw.groupby(target['cell']['key'] + ['area_round'])
    names = [s['name'] for specs in features['base'].values() for s in specs]
    checks = 0
    for i, regime in enumerate(regimes):
        start, end, period = bounds(regime, target, i, len(regimes))
        for method in ['window', 'slope']:
            table = pd.read_csv(out / f'{regime["id"]}_{method}_v1.csv.xz', dtype={'번지': str})
            assert not table.cell_id.duplicated().any()
            assert table['연식_개월_a'].ge(0).all()
            assert np.isfinite(table[['y', 'weight']]).all().all()
            assert table.weight.gt(0).all()
            assert table.feature_asof.eq(str(pd.Period(regime['a'], 'M').end_time.date())).all()
            for name in names:
                assert table[name].isna().astype(int).eq(table[name+'_missing']).all()
            # Evenly spaced samples are deterministic and do not inspect model performance.
            for pos in np.linspace(0, len(table)-1, min(20, len(table)), dtype=int):
                row = table.iloc[pos]
                key = tuple(row[k] for k in target['cell']['key']) + (row['전용면적'],)
                g = groups.get_group(key)
                if method == 'window':
                    s, e = [g[g.period.between(*interval)] for interval in [start, end]]
                    log_a, log_b = np.log(idx.loc[regime['a']]), np.log(idx.loc[regime['b']])
                    price_a = np.median(np.log(s.price/s.area) - np.log(s['index']) + log_a)
                    price_b = np.median(np.log(e.price/e.area) - np.log(e['index']) + log_b)
                    expected_y = (price_b-price_a-(log_b-log_a))*100
                    expected_w = min(len(s), len(e))
                else:
                    g = g[g.period.between(*period)]
                    x = g.period.to_numpy()-g.period.min()
                    z = (np.log(g.price)-np.log(g['index']))*100
                    expected_y = np.polyfit(x, z, 1)[0]*regime['L']
                    expected_w = len(g)*(g.period.max()-g.period.min())
                assert np.isclose(row.y, expected_y, atol=1e-7, rtol=1e-8), (regime['id'], method, row.cell_id)
                assert row.weight == expected_w
                checks += 1
    result = {'tables_checked': len(regimes)*2, 'independent_targets_checked': checks,
              'source_rows_reconciled': n, 'passed': True}
    (out / 'verification_v1.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
