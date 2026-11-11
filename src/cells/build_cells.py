"""Build reproducible cell x regime tables, without training or opening a test split.

Run from the repository root: python src/cells/build_cells.py
Input transactions are immutable; every exclusion is recorded separately.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/pipeline'))
from build_access_features_v2 import hav, stations
from build_korean_dataset_v2 import schools


def read_config(name):
    return yaml.safe_load((ROOT / 'config' / f'{name}.yaml').read_text(encoding='utf-8'))


def month(value):
    return pd.Period(value, freq='M').ordinal


def bounds(regime, target, position, total):
    a, b = month(regime['a']), month(regime['b'])
    w = target['window']
    k = w['half_width_months_by_regime'].get(regime['id'], w['half_width_months'])
    start, end = (a-k, a+k), (b-k, b+k)
    if position == 0:
        start = tuple(map(month, w['first_start_window']))
    if position == total-1:
        end = tuple(map(month, w['last_end_window']))
    if start[1] >= end[0]:
        raise ValueError('Start/end windows overlap')
    return start, end, (a-k, b+k)


def mode(series):
    counts = series.replace('', pd.NA).dropna().value_counts()
    return sorted(counts[counts == counts.max()].index)[0] if len(counts) else np.nan


def prepare(raw, target, index):
    d = raw.copy()
    if d['거래ID'].duplicated().any():
        raise ValueError('Duplicate transaction IDs')
    keys = target['cell']['key']
    if d[keys].eq('').any().any():
        raise ValueError('Missing cell identity')
    d['m'] = pd.to_datetime(d['계약일'], errors='raise').dt.to_period('M').astype('int64')
    for c in ['전용면적(㎡)', '거래금액_만원', '층', '위도', '경도']:
        d[c] = pd.to_numeric(d[c].str.replace(',', '', regex=False), errors='coerce')
    if not (d['전용면적(㎡)'].gt(0) & d['거래금액_만원'].gt(0)).all():
        raise ValueError('Nonpositive/missing area or price')
    unit = target['cell']['area_rounding_m2']
    d['area_cell'] = np.round(d['전용면적(㎡)'] / unit) * unit
    # JSON tuples avoid ambiguous delimiter collisions in Korean apartment names.
    identities = d[keys].drop_duplicates().copy()
    identities['complex_id'] = identities.apply(
        lambda r: json.dumps(r.tolist(), ensure_ascii=False, separators=(',', ':')), axis=1)
    d = d.merge(identities, on=keys, how='left', validate='many_to_one')
    d['cell_id'] = d['complex_id'] + '#' + d['area_cell'].astype(str)
    d['dong_id'] = d['자치구'] + '|' + d['법정동']
    mapped = d['m'].map(index)
    if mapped.isna().any() or not mapped.gt(0).all():
        raise ValueError('Index missing/nonpositive for a transaction month')
    d['z_unit'] = np.log(d['거래금액_만원'] / d['전용면적(㎡)']) - np.log(mapped)
    d['z_total'] = (np.log(d['거래금액_만원']) - np.log(mapped)) * 100
    d['exclusion_reason'] = ''
    missing = d[['위도', '경도']].isna().any(axis=1)
    pre = d['연식_기준'].eq('준공 전 거래')
    d.loc[missing, 'exclusion_reason'] = '좌표 없음;'
    d.loc[pre, 'exclusion_reason'] += '준공 전 거래;'
    d['exclusion_reason'] = d['exclusion_reason'].str.rstrip(';')
    return d


def window_target(d, start, end):
    def aggregate(interval):
        return d[d.m.between(*interval)].groupby('cell_id').agg(
            z=('z_unit', 'median'), n=('m', 'size'), floor=('층', 'median'))
    s, e = aggregate(start), aggregate(end)
    result = s.join(e, how='outer', lsuffix='_start', rsuffix='_end')
    result['target_exclusion'] = np.select(
        [result.n_start.isna(), result.n_end.isna()],
        ['시작 창 거래 없음', '끝 창 거래 없음'], default='')
    # Boundary-index constants cancel algebraically in relative log changes.
    result['y'] = (result.z_end - result.z_start) * 100
    result['weight'] = result[['n_start', 'n_end']].min(axis=1)
    result['층'] = result.floor_start
    return result


def slope_target(d, period, length, target):
    w = d[d.m.between(*period)].copy()
    w['t'] = w.m - period[0]
    w['tt'] = w.t**2
    w['tz'] = w.t * w.z_total
    g = w.groupby('cell_id').agg(n=('t', 'size'), t=('t', 'sum'),
        tt=('tt', 'sum'), z=('z_total', 'sum'), tz=('tz', 'sum'),
        first_month=('m', 'min'), last_month=('m', 'max'), 층=('층', 'median'))
    g['span_months'] = g.last_month - g.first_month
    denom = g.tt - g.t**2 / g.n
    g['slope_per_month'] = (g.tz - g.t*g.z/g.n) / denom.where(denom > 0)
    g['y'] = g.slope_per_month * length
    g['weight'] = g.n * g.span_months
    g['target_exclusion'] = np.select(
        [g.n < target['slope']['min_trades'],
         g.span_months < target['slope']['min_span_months']],
        ['거래 수 부족', '거래 간격 부족'], default='')
    return g


def access_at(points, date, radius):
    """Reuse the repository's facility histories at the regime start month end."""
    result = pd.DataFrame(index=points.index)
    facilities = {'stations': stations()}
    facilities.update({level: schools(level) for level in ['초등학교', '중학교', '고등학교']})
    counts = pd.Series(0, index=points.index)
    for kind, all_rows in facilities.items():
        active = [r for r in all_rows if r['open'] <= date < r['close']]
        if not active:
            raise ValueError(f'No active facilities: {kind}, {date}')
        distances = hav(points['위도'].to_numpy(float), points['경도'].to_numpy(float),
            np.array([r['lat'] for r in active]), np.array([r['lon'] for r in active]))
        if kind == 'stations':
            result['최근접_역_거리'] = distances.min(axis=1)
            lines = np.array([r['line'] for r in active])
            result['1km내_노선수'] = sum((distances[:, lines == line] <= radius).any(axis=1)
                for line in sorted(set(lines)))
        else:
            counts += (distances <= radius).sum(axis=1)
            if kind == '초등학교':
                result['최근접_초등학교_거리'] = distances.min(axis=1)
    result['1km내_학교수'] = counts
    return result


def static_features(d, feature_cfg):
    specs = [s for group in feature_cfg['base'].values() for s in group]
    columns = {s['column'] for s in specs if 'column' in s}
    columns.update(['사용승인일', '건축년도', '좌표기준', '세대수_기준'])
    columns -= {'전용면적(㎡)', '최근접_역_거리(m)', '1km내_노선수',
                '최근접_초등학교_거리(m)', '1km내_학교수', '자치구'}
    grouped = d.groupby('complex_id', sort=True)
    static = grouped[sorted(columns)].agg(mode)
    # Select an observed coordinate pair, never independent latitude/longitude medians.
    pairs = d[['complex_id', '위도', '경도']].dropna().value_counts().rename('n').reset_index()
    pairs = pairs.sort_values(['complex_id', 'n', '위도', '경도'], ascending=[True, False, True, True])
    static = static.join(pairs.drop_duplicates('complex_id').set_index('complex_id')[['위도', '경도']])
    conflicts = grouped[sorted(columns)].nunique(dropna=False)
    conflicts = conflicts.where(conflicts > 1).stack().rename('distinct_values').reset_index()
    conflicts.columns = ['complex_id', 'column', 'distinct_values']
    return static, conflicts


def features_at(d, static, regime, cfg, feature_cfg):
    a = month(regime['a'])
    date = str(pd.Period(regime['a'], 'M').end_time.date())
    f = static.copy()
    approval = pd.to_datetime(f['사용승인일'], errors='coerce')
    fallback = pd.to_numeric(f['건축년도'], errors='coerce')
    approval_month = approval.dt.year * 12 + approval.dt.month - 1 - 1970*12
    fallback_month = fallback * 12 + cfg['fallback_approval_month'] - 1 - 1970*12
    f['연식_개월_a'] = a - approval_month.fillna(fallback_month)
    f['재건축연한'] = (f['연식_개월_a'] >= cfg['reconstruction_age_months']).astype(float)
    f.loc[f['연식_개월_a'].isna(), '재건축연한'] = np.nan
    f['age_basis'] = np.where(approval.notna(), '사용승인일', '건축년도 7월 가정')
    f = f.join(access_at(f[['위도', '경도']], date, cfg['radius_m']))
    identity = d[['cell_id', 'complex_id', 'dong_id', '자치구', '법정동', '번지', '단지명', 'area_cell']].drop_duplicates('cell_id').set_index('cell_id')
    f = identity.join(f, on='complex_id')
    f['전용면적'] = f.area_cell
    f['feature_asof'] = date
    for group in feature_cfg['base'].values():
        for spec in group:
            name = spec['name']
            if name in f and spec.get('type') != 'category':
                f[name] = pd.to_numeric(f[name], errors='coerce')
    f['cell_exclusion'] = np.select(
        [f['연식_개월_a'].isna(), f['연식_개월_a'] < 0],
        ['국면 시작 연식 계산 불가', '국면 시작 시 준공 전'], default='')
    return f


def controls(d, features, regime, previous, target, cfg, start, tohuh):
    a = month(regime['a'])
    cc = cfg['controls']
    lower, upper = cc['price_lookback_months']
    end = min(a-upper, start[0]-1)
    pre = d[d.m.between(a-lower, end)]
    med = pre.groupby('cell_id')['거래금액_만원'].median().reindex(features.index)
    low, high = cc['price_cutoffs_manwon']
    out = pd.DataFrame(index=features.index)
    out['가격대'] = pd.cut(med, [-np.inf, low, high, np.inf], labels=['9억 이하', '9~15억', '15억 초과'])
    out['price_history_n'] = pre.groupby('cell_id').size().reindex(out.index).fillna(0)
    history = d[d.m.between(start[0]-cc['turnover_months'], start[0]-1)]
    counts = history.groupby('cell_id').size().reindex(out.index).fillna(0)
    out['회전율'] = counts / features['세대수'].where(features['세대수'] > 0)
    # A zero count is not zero turnover if the full observation window is unavailable.
    out['turnover_history_complete'] = start[0]-cc['turnover_months'] >= d.m.min()
    out.loc[~out.turnover_history_complete, '회전율'] = np.nan
    out['직전국면_y'] = np.nan
    if previous:
        ps, pe = previous
        pe = (pe[0], min(pe[1], start[0]-1))
        if pe[0] <= pe[1] and ps[1] < pe[0]:
            lag = window_target(d, ps, pe)
            lag = lag[lag.target_exclusion.eq('')]
            out['직전국면_y'] = lag.y.reindex(out.index)
    key = (features['자치구'].radd('서울특별시 ') + ' ' + features['법정동'])
    keys = pd.DataFrame({'시군구': key, '번지': features['번지'], '단지명': features['단지명']})
    t = tohuh[tohuh['국면'].eq(regime['name'])]
    joined = keys.reset_index().merge(t[['시군구', '번지', '단지명', '기준일', '토지거래허가구역']],
        on=['시군구', '번지', '단지명'], how='left', validate='many_to_one').set_index('cell_id')
    if joined['기준일'].dropna().ne(features.feature_asof.iloc[0]).any():
        raise ValueError('Land-permit as-of date mismatch')
    out['토지거래허가구역_a'] = pd.to_numeric(joined['토지거래허가구역'], errors='coerce').reindex(out.index)
    return out


def write_csv(frame, path, index=False):
    frame.to_csv(path, index=index, encoding='utf-8-sig', lineterminator='\n', float_format='%.10g')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg, target, feature_cfg = [read_config(n) for n in ['cells', 'target', 'features']]
    regimes = read_config('regime')['regimes']
    supported = {'slope_weight': 'n_times_span_months', 'window_price': 'price_per_m2',
        'slope_price': 'total_price', 'area_rounding': 'half_even', 'feature_date': 'start_month_end',
        'static_snapshot': 'most_frequent_nonmissing', 'not_built_at_start': 'exclude_cell'}
    if any(cfg[k] != v for k, v in supported.items()):
        raise ValueError('Unsupported target implementation settings')
    if cfg['controls']['previous_y'] != 'previous_window_truncated_before_current_start_window':
        raise ValueError('Unsupported previous-y setting')
    if target['window']['statistic'] != 'median':
        raise ValueError('Unsupported window statistic')
    out = ROOT / cfg['output']
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / cfg['source'], dtype=str, keep_default_na=False)
    index = pd.read_csv(ROOT / 'data/regime/seoul_index_2017_2026.csv', index_col=0)[target['index_for_time_adjustment']]
    index.index = [month(x) for x in index.index]
    d = prepare(raw, target, index)
    excluded = d[d.exclusion_reason.ne('')]
    write_csv(excluded[['거래ID', 'cell_id', '계약일', 'exclusion_reason']], out / 'excluded_trades_v1.csv')
    clean = d[d.exclusion_reason.eq('')].copy()
    static, conflicts = static_features(clean, feature_cfg)
    write_csv(conflicts, out / 'static_conflicts_v1.csv')
    tohuh = pd.read_csv(ROOT / feature_cfg['controls'][0]['source'], dtype=str, keep_default_na=False)
    summaries, exclusions, missing = [], [], []
    previous = None
    base_names = [s['name'] for group in feature_cfg['base'].values() for s in group]
    for i, r in enumerate(regimes):
        start, end, period = bounds(r, target, i, len(regimes))
        f = features_at(clean, static, r, cfg, feature_cfg)
        f = f.join(controls(clean, f, r, previous, target, cfg, start, tohuh))
        for method in ['window', 'slope']:
            compute = (lambda frame: window_target(frame, start, end)) if method == 'window' else (
                lambda frame: slope_target(frame, period, r['L'], target))
            unfiltered, candidates = compute(d), compute(clean)
            unfiltered_n = int(unfiltered.target_exclusion.eq('').sum())
            count = int(candidates.target_exclusion.eq('').sum())
            valid = candidates[candidates.target_exclusion.eq('')].join(f)
            dropped = candidates[candidates.target_exclusion.ne('')][['target_exclusion']].rename(columns={'target_exclusion': 'reason'})
            extra = valid[valid.cell_exclusion.ne('')][['cell_exclusion']].rename(columns={'cell_exclusion': 'reason'})
            # Include cells removed completely by transaction exclusions.
            removed_ids = unfiltered.index[unfiltered.target_exclusion.eq('')].difference(candidates.index)
            removed = pd.DataFrame({'reason': '제외 거래만 존재'}, index=removed_ids)
            audit = pd.concat([dropped, extra, removed]).reset_index().rename(columns={'index': 'cell_id'})
            audit.insert(0, 'method', method); audit.insert(0, 'regime', r['id'])
            exclusions.append(audit)
            valid = valid[valid.cell_exclusion.eq('')].copy()
            valid['regime'] = r['id']; valid['method'] = method
            for name in base_names:
                valid[name + '_missing'] = valid[name].isna().astype(int)
                missing.append({'regime': r['id'], 'method': method, 'feature': name,
                    'missing_n': int(valid[name].isna().sum()), 'cells': len(valid)})
            if not np.isfinite(valid[['y', 'weight']]).all().all() or not valid.weight.gt(0).all():
                raise ValueError('Invalid target/weight')
            if valid.index.duplicated().any():
                raise ValueError('Duplicate cells')
            write_csv(valid.sort_index().reset_index(), out / f'{r["id"]}_{method}_v1.csv.xz')
            summaries.append({'regime': r['id'], 'name': r['name'], 'method': method,
                'expected_v4': target['expected_cells'][method][r['id']], 'unfiltered_cells': unfiltered_n,
                'after_trade_exclusions': count, 'not_built_at_start': len(extra), 'cells': len(valid),
                'complexes': valid.complex_id.nunique(), 'dongs': valid.dong_id.nunique(),
                'y_mean': valid.y.mean(), 'y_median': valid.y.median(), 'y_p01': valid.y.quantile(.01),
                'y_p99': valid.y.quantile(.99), 'y_min': valid.y.min(), 'y_max': valid.y.max(),
                'weight_sum': valid.weight.sum()})
            print(r['id'], method, 'unfiltered', unfiltered_n, 'after trade exclusions', count, 'final', len(valid), flush=True)
        previous = (start, end)
    summary = pd.DataFrame(summaries)
    write_csv(summary, out / 'summary_v1.csv')
    write_csv(pd.concat(exclusions, ignore_index=True), out / 'excluded_cells_v1.csv.xz')
    write_csv(pd.DataFrame(missing), out / 'missing_features_v1.csv')
    inputs = [ROOT / cfg['source'], ROOT / 'data/regime/seoul_index_2017_2026.csv',
        ROOT / feature_cfg['controls'][0]['source']]
    inputs += sorted((ROOT / 'config').glob('*.yaml')) + sorted((ROOT / 'config/access_history').glob('*.csv'))
    inputs += [ROOT / 'data/raw/kric_stations_bbox_20260630.csv'] + sorted((ROOT / 'data/raw').glob('schools_page*.json'))
    provenance = {'source_rows': len(raw), 'source_columns': len(raw.columns),
        'excluded_trades': len(excluded), 'exclusion_reasons': excluded.exclusion_reason.value_counts().to_dict(),
        'retained_trades': len(clean), 'static_conflicts': len(conflicts),
        'unfiltered_counts_match_expected': bool(summary.unfiltered_cells.eq(summary.expected_v4).all()),
        'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
        'model_run': False, 'test_split_created': False,
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}}
    (out / 'provenance_v1.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in provenance.items() if k != 'input_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
