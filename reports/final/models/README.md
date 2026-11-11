# 국면별 모델 학습·평가 v1

## 고정한 실험

2026-11-11 모델링 진행 요청에 따라 기존 `config/split.yaml`, `config/models.yaml`의
시드와 격자를 변경 없이 채택했다. 구현 세부 규칙은 `config/training.yaml`에 기록했다.

- 전체 319개 법정동의 정렬된 목록을 시드 20261103의 NumPy RNG로 섞고,
  ceil(319×0.2)=64개를 공통 test로 고정했다. 나머지는 255개다.
- 국면×방식별 학습 표에서 `GroupKFold(5)`를 만든다. 표가 같으면 모든 모델의 fold도 같다.
  국면 간 학습 fold는 표본 수 차이 때문에 달라질 수 있지만 test 동 목록은 공통이다.
- 기본 15개 특성과 각각의 결측 표시 15개만 X로 사용한다. 식별자·가격·목표값·통제변수는 X에 넣지 않는다.
- Ridge와 랜덤포레스트의 숫자 결측은 학습 fold 중앙값으로 대체한다. 전부 결측인 열은
  열을 유지하고 0으로 처리하며 결측 표시가 함께 들어간다. 범주는 결측 토큰과 원핫 인코딩을 사용한다.
  Ridge 숫자에는 학습 fold에서 적합한 표준화를 적용한다.
- CatBoost의 숫자 결측은 NaN, 범주 결측은 문자열 토큰을 사용한다.
- 표본 가중치는 분석표에 저장된 원래 값을 사용한다. 전처리 통계는 비가중이며 모델 학습·평가에는 가중치를 적용한다.

## 모델 선택

기준선 b0는 0, b1은 학습 데이터 자치구별 가중평균 y다.
학습에 없는 자치구는 학습 전체 가중평균을 사용한다.

Ridge 6개, 랜덤포레스트 18개, CatBoost 12개 후보를 각 5개 fold로 검증한다.
후보 선택 기준은 전체 OOF 예측의 가중 RMSE이며, 동점이면 설정 격자의 앞 후보를 선택한다.
CatBoost 조기 종료에는 해당 학습 내부 검증 fold만 사용한다. 최종 학습 반복 수는
선택된 후보의 fold별 best iteration 수 중앙값(정수)을 쓰며 test는 eval_set에 넣지 않는다.
선택한 모델을 해당 표의 학습 법정동 전체에 다시 적합한다. 20%는 동 수 기준이므로 셀 수의 test 비율은 표마다 다르다.

기준선도 같은 학습 내부 fold에서 OOF 예측을 만든다. 각 표에서 기준선 2개와 학습 모델 3개 중
OOF 가중 RMSE가 가장 작은 모델을 최종 평가 전에 고정한다(`cv_winners_v1.csv`).
동점이면 b0, b1, Ridge, 랜덤포레스트, CatBoost 순이다. test 결과가 다르더라도 선택을 바꾸지 않는다.

## 최종 평가

모든 표의 모델 설정을 확정한 뒤 test 예측을 한 번 저장한다. 저장 결과를 보고
격자·분할·특성·이상치 규칙을 바꾸지 않는다. 평가 시작 표시가 있으면 중복 평가를 거부하고,
평가 결과 파일이 있으면 재튜닝도 거부한다. 평가 도중 중단된 경우에는 재튜닝하지 않고 저장된 예측·오류부터 점검한다.

- RMSE·MAE: 거래 신뢰도 가중치를 적용한 오차, 단위는 y와 같은 로그 변화 차이×100.
- Spearman: y와 예측의 평균 순위에 대한 가중 Pearson 상관. 상수 예측이면 정의 불가이므로 빈 값이다.
- 상하위 20% 차이: **셀 수 기준** 예측 상위/하위 20%의 가중 실제 y 평균 차이.
  경계에서 동점인 셀은 같은 비율로 포함한다. 따라서 상수 예측의 차이는 0이다.
- 95% 구간: test의 단지를 복원 추출해 그 단지의 모든 면적 셀을 함께 가져오는 1,000회 부트스트랩.
  동일 추출을 모든 모델에 적용하고, b1 대비 RMSE 차이의 짝지은 구간도 계산한다.
  이 구간은 **적합된 모델과 고정된 동 분할에 조건부**이며 재학습·튜닝의 불확실성을 포함하지 않는다.
- `eligible_for_interpretation`: test RMSE가 b1보다 작은 학습 모델 표시.
  이는 이후 특성 해석 단계의 조건이다. 이 값으로 재튜닝하거나 예측을 바꾸지 않는다.

## 파일과 실행

- `dong_assignments_v1.csv`, `../split_test_dongs.csv`: 공통 동 분할.
- `cell_splits_v1.csv.xz`: 모든 74,415개 셀×국면×방식 행의 train/test와 학습 내부 fold.
- `split_summary_v1.csv`, `split_manifest_v1.json`: 표본 수, 입력·설정·코드 해시.
- `cv/`: 후보별 5개 검증 점수와 통합 점수. CatBoost의 fold별 반복 수도 저장한다.
- `selections/`: 선택한 설정, 학습 표본 수, 모델 파일 해시.
- `cv_comparison_v1.csv`, `cv_winners_v1.csv`: 기준선을 포함한 학습 내부 비교와 test 이전의 모델 선택.
- `oof/`: 선택 모델의 학습 내부 검증 예측.
- `predictions/`: 고정된 모델의 test 예측.
- `test_metrics_v1.csv`, `test_intervals_v1.csv`: 성능 및 신뢰구간.
- `verification_v1.json`: 분할 누출·예측 행·오차 계산 검증.

```bash
python src/model/train.py split
python src/model/train.py train
python tools/freeze_model_choices.py
python src/model/train.py evaluate
python tools/verify_models.py
python -m pytest -q
python tools/verify_final.py
```

학습 모델 파일은 크기가 커 Git 제외 경로 `results/models_v1/`에 저장한다.
저장소에는 설정·코드·예측·점수·해시를 보관한다. 완료된 실험을 새 환경에 복제했을 때는
`python tools/refit_models.py`로 **저장된 설정만** 사용해 학습용 데이터에 다시 적합한다.
이 명령은 튜닝이나 test 평가를 다시 하지 않는다. 환경 차이에 따른 직렬화 파일 해시 차이는
`results/models_v1/refit_manifest.json`에 기록한다.
완료된 학습은 동일한 입력·코드와 모델 파일 해시가 확인될 때만 이어서 실행할 수 있다.
이미 test 평가를 완료한 실험에서는 저장된 점수를 `tools/verify_models.py`로 검증한다.

이 단계는 기본 모델 비교까지다. 계수·SHAP, 특성 유형표와 확장 통제변수 모델은 후속 단계다.
기울기 방식은 강건성 비교용이며, 큰 y를 임의로 삭제하지 않았다.
