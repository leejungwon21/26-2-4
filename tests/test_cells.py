"""Analytical fixtures check targets and time boundaries independently of real outcomes."""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_cells', ROOT / 'src/cells/build_cells.py')
cells = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cells)


def test_window_market_matching_returns_zero_and_outperformance_positive():
    # Prices rise 20% with the market, or 32% (10% excess multiplicative growth).
    d = pd.DataFrame({'cell_id': ['market', 'market', 'winner', 'winner', 'one_sided'],
        'm': [0, 12, 0, 12, 0], '층': [1, 9, 3, 10, 2],
        'z_unit': np.log(np.array([100, 120, 100, 132, 100])/np.array([100, 120, 100, 120, 100]))})
    result = cells.window_target(d, (0, 0), (12, 12))
    assert result.loc['market', 'y'] == pytest.approx(0)
    assert result.loc['winner', 'y'] == pytest.approx(100*np.log(1.1))
    assert result.loc['winner', '층'] == 3  # End-window floor must not enter.
    assert result.loc['winner', 'weight'] == 1
    assert result.loc['one_sided', 'target_exclusion'] == '끝 창 거래 없음'


def test_slope_matches_independent_ols_and_checks_span():
    d = pd.DataFrame({'cell_id': ['a']*3 + ['short']*2 + ['single'],
        'm': [10, 14, 19, 10, 15, 10], 'z_total': [4, 5, 9.4, 1, 2, 1], '층': [1]*6})
    target = cells.read_config('target')
    result = cells.slope_target(d, (10, 20), 15, target)
    assert result.loc['a', 'y'] == pytest.approx(np.polyfit([0, 4, 9], [4, 5, 9.4], 1)[0]*15)
    assert result.loc['a', 'weight'] == 3*9
    assert result.loc['short', 'target_exclusion'] == '거래 간격 부족'
    assert result.loc['single', 'target_exclusion'] == '거래 수 부족'
    shifted = d.copy(); shifted['z_total'] += 1000
    assert cells.slope_target(shifted, (10, 20), 15, target).loc['a', 'y'] == pytest.approx(result.loc['a', 'y'])


def test_config_windows_do_not_overlap_and_special_endpoints_apply():
    target = cells.read_config('target')
    regimes = cells.read_config('regime')['regimes']
    for i, r in enumerate(regimes):
        start, end, _ = cells.bounds(r, target, i, len(regimes))
        assert start[1] < end[0]
    assert cells.bounds(regimes[0], target, 0, 6)[0] == tuple(map(cells.month, ['2017-01', '2017-04']))
    assert cells.bounds(regimes[-1], target, 5, 6)[1] == tuple(map(cells.month, ['2026-05', '2026-10']))
    start, end, _ = cells.bounds(regimes[1], target, 1, 6)
    assert end[0]-start[1] == 2


def test_access_uses_only_open_facilities_and_deduplicates_lines(monkeypatch):
    def facility(open_date, close_date='9999-12-31', lat=37.5, line='L1'):
        return dict(open=open_date, close=close_date, lat=lat, lon=127, line=line)
    monkeypatch.setattr(cells, 'stations', lambda: [facility('2000-01-01', lat=37.51),
        facility('2000-01-01', lat=37.51), facility('2020-02-01', line='L2')])
    monkeypatch.setattr(cells, 'schools', lambda level: [facility('2000-01-01'),
        facility('2020-02-01'), facility('2000-01-01', close_date='2020-01-31')])
    pts = pd.DataFrame({'위도': [37.5], '경도': [127.]})
    before = cells.access_at(pts, '2020-01-31', 2000)
    after = cells.access_at(pts, '2020-02-01', 2000)
    assert before.loc[0, '최근접_역_거리'] > 1000
    assert before.loc[0, '1km내_노선수'] == 1
    assert after.loc[0, '1km내_노선수'] == 2
    assert before.loc[0, '1km내_학교수'] == 3
    assert after.loc[0, '1km내_학교수'] == 6


def test_controls_exclude_target_window_prices_and_partial_history():
    cfg = cells.read_config('cells'); target = cells.read_config('target')
    a = cells.month('2020-01')
    d = pd.DataFrame({'cell_id': ['a']*4, 'm': [a-15, a-12, a-4, a-3],
                      '거래금액_만원': [50000, 80000, 100000, 9999999]})
    f = pd.DataFrame({'자치구': ['강남구'], '법정동': ['역삼동'], '번지': ['1'],
        '단지명': ['A'], '세대수': [100], 'feature_asof': ['2020-01-31']},
        index=pd.Index(['a'], name='cell_id'))
    t = pd.DataFrame({'시군구': ['서울특별시 강남구 역삼동'], '번지': ['1'], '단지명': ['A'],
        '국면': ['test'], '기준일': ['2020-01-31'], '토지거래허가구역': ['1']})
    result = cells.controls(d, f, {'a': '2020-01', 'name': 'test'}, None, target, cfg, (a-3, a+3), t)
    assert result.loc['a', '가격대'] == '9억 이하'  # Median is exactly 90,000.
    assert result.loc['a', '회전율'] == pytest.approx(.03)
    assert pd.isna(result.loc['a', '직전국면_y'])
    assert result.loc['a', '토지거래허가구역_a'] == 1
    partial = cells.controls(d[d.m >= a-12], f, {'a': '2020-01', 'name': 'test'}, None, target, cfg, (a-3, a+3), t)
    assert pd.isna(partial.loc['a', '회전율'])


def test_age_recomputed_at_start_and_future_buildings_flagged(monkeypatch):
    monkeypatch.setattr(cells, 'access_at', lambda points, date, radius: pd.DataFrame(index=points.index))
    static = pd.DataFrame({'사용승인일': ['2020-02-01', ''], '건축년도': ['2020', '1989'],
        '위도': [37.5, 37.5], '경도': [127., 127.]}, index=['future', 'old'])
    d = pd.DataFrame({'cell_id': ['a', 'b'], 'complex_id': ['future', 'old'], 'dong_id': ['x', 'x'],
        '자치구': ['강남구']*2, '법정동': ['역삼동']*2, '번지': ['1', '2'], '단지명': ['A', 'B'], 'area_cell': [84, 59]})
    f = cells.features_at(d, static, {'a': '2020-01'}, cells.read_config('cells'), cells.read_config('features'))
    assert f.loc['a', '연식_개월_a'] == -1
    assert f.loc['a', 'cell_exclusion'] == '국면 시작 시 준공 전'
    assert f.loc['b', '연식_개월_a'] == 366
    assert f.loc['b', '재건축연한'] == 1


def test_prepare_preserves_rows_and_reports_both_exclusions():
    raw = pd.DataFrame({'거래ID': ['1', '2'], '자치구': ['구']*2, '법정동': ['동']*2,
        '번지': ['1']*2, '단지명': ['A']*2, '계약일': ['2020-01-01']*2,
        '전용면적(㎡)': ['84.5', '85.5'], '거래금액_만원': ['10,000']*2,
        '층': ['1']*2, '위도': ['37.5', ''], '경도': ['127', ''], '연식_기준': ['', '준공 전 거래']})
    result = cells.prepare(raw, cells.read_config('target'), pd.Series({cells.month('2020-01'): 100}))
    assert len(result) == 2
    assert result.area_cell.tolist() == [84, 86]
    assert '좌표 없음' in result.loc[1, 'exclusion_reason']
    assert '준공 전 거래' in result.loc[1, 'exclusion_reason']
    with pytest.raises(ValueError, match='Index missing'):
        cells.prepare(raw, cells.read_config('target'), pd.Series({cells.month('2019-01'): 100}))
