"""Render verified model scores as a compact Markdown result table."""
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/model'))
from train import config


def main():
    output = ROOT/config('training')['output']
    d = pd.read_csv(output/'test_metrics_v1.csv')
    ci = pd.read_csv(output/'test_intervals_v1.csv')
    winners = pd.read_csv(output/'cv_winners_v1.csv').set_index('table')['selected_model']
    labels = {r['id']: r['name'] for r in config('regime')['regimes']}
    lines = ['# 기본 모델 평가 결과 v1', '',
        '법정동 test 64개를 모든 국면에 공통 적용했다. 모델 설정은 학습 내부 교차검증으로 확정한 뒤 test를 평가했다.', '',
        '## Test 가중 RMSE', '',
        '단위: 서울 대비 상대 로그 가격 변화 ×100. 같은 국면·방식 안에서 작을수록 좋다. `해석 대상`은 b1보다 RMSE가 낮은 모델이다.',
        '창·기울기 방식은 목표값 정의와 표본이 달라 RMSE 크기만으로 두 방식의 우열을 비교하지 않는다.', '',
        '| 국면 | 방식 | test 셀 | b0 | b1(자치구) | Ridge | RF | CatBoost | CV 선택 | 해석 대상 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in config('regime')['regimes']:
        for method in config('training')['methods']:
            key = f'{r["id"]}_{method}'; sub = d[d.table.eq(key)].set_index('model')
            eligible = sub.index[sub.eligible_for_interpretation].tolist()
            values = [f'{sub.loc[m, "RMSE"]:.3f}' for m in ['b0', 'b1', 'ridge', 'random_forest', 'catboost']]
            lines.append(f'| {labels[r["id"]]} | {method} | {int(sub.test_cells.iloc[0]):,} | '+ ' | '.join(values) + ' | ' + winners.loc[key] + ' | ' + (', '.join(eligible) or '없음') + ' |')
    lines += ['', '## b1 대비 RMSE 차이의 95% 구간', '',
        '음수는 b1보다 오차가 작다는 뜻이다. 구간이 0을 포함하면 개선의 불확실성이 남는다.', '',
        '| 국면·방식 | 모델 | RMSE 차이 | 95% 구간 |', '|---|---|---:|---|']
    for row in d[d.model.isin(['ridge', 'random_forest', 'catboost'])].itertuples():
        interval = ci[ci.table.eq(row.table)&ci.model.eq(row.model)&ci.metric.eq('rmse_minus_b1')].iloc[0]
        lines.append(f'| {row.table} | {row.model} | {row.rmse_minus_b1:.3f} | [{interval.low:.3f}, {interval.high:.3f}] |')
    lines += ['', '## 해석 범위', '',
        '- 이 결과는 고정된 모델의 동네 외부 평가다. 새로운 시기나 새로운 시장 국면에 대한 예측력을 검증한 것은 아니다.',
        '- 기본 15개 특성과 결측 표시만 사용했다. 통제변수 확장 모델·SHAP·특성 유형표는 아직 계산하지 않았다.',
        '- 부트스트랩 구간은 단지를 묶어 1,000회 재표집한 결과이며, 모델 재학습의 불확실성은 포함하지 않는다.',
        '- 모든 MAE·순위상관·상하위 20% 차이와 구간은 `test_metrics_v1.csv`, `test_intervals_v1.csv`에 있다.',
        '- 상수 예측인 b0의 순위상관은 정의되지 않으므로 빈 값이다. 상하위 차이는 동점 처리 후 0이다.',
        '- 큰 오차를 보고 test 표본을 제거하거나 모델을 다시 튜닝하지 않았다.', '']
    (output/'results_v1.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
