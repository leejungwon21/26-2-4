# 기본 모델 평가 결과 v1

법정동 test 64개를 모든 국면에 공통 적용했다. 모델 설정은 학습 내부 교차검증으로 확정한 뒤 test를 평가했다.

## Test 가중 RMSE

단위: 서울 대비 상대 로그 가격 변화 ×100. 같은 국면·방식 안에서 작을수록 좋다. `해석 대상`은 b1보다 RMSE가 낮은 모델이다.
창·기울기 방식은 목표값 정의와 표본이 달라 RMSE 크기만으로 두 방식의 우열을 비교하지 않는다.

| 국면 | 방식 | test 셀 | b0 | b1(자치구) | Ridge | RF | CatBoost | CV 선택 | 해석 대상 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 상승기① | window | 1,236 | 12.841 | 13.137 | 10.630 | 9.863 | 10.251 | catboost | ridge, random_forest, catboost |
| 상승기① | slope | 2,304 | 14.595 | 14.004 | 11.520 | 10.710 | 10.903 | catboost | ridge, random_forest, catboost |
| 조정기 | window | 984 | 6.633 | 6.732 | 6.199 | 6.257 | 6.199 | catboost | ridge, random_forest, catboost |
| 조정기 | slope | 938 | 4.508 | 4.519 | 4.205 | 4.247 | 4.208 | catboost | ridge, random_forest, catboost |
| 상승기② | window | 996 | 17.810 | 16.920 | 13.111 | 9.705 | 9.253 | catboost | ridge, random_forest, catboost |
| 상승기② | slope | 2,620 | 19.257 | 18.236 | 15.077 | 11.819 | 11.707 | catboost | ridge, random_forest, catboost |
| 하락기 | window | 658 | 12.477 | 12.686 | 11.354 | 9.284 | 9.264 | catboost | ridge, random_forest, catboost |
| 하락기 | slope | 976 | 12.852 | 12.778 | 11.693 | 10.080 | 9.985 | catboost | ridge, random_forest, catboost |
| 회복기 | window | 735 | 9.058 | 8.790 | 7.672 | 7.043 | 7.043 | catboost | ridge, random_forest, catboost |
| 회복기 | slope | 1,396 | 9.940 | 9.473 | 8.646 | 8.301 | 7.965 | catboost | ridge, random_forest, catboost |
| 상승기③ | window | 1,203 | 15.524 | 15.274 | 13.129 | 10.804 | 10.687 | catboost | ridge, random_forest, catboost |
| 상승기③ | slope | 2,217 | 18.356 | 17.062 | 14.939 | 12.145 | 12.840 | catboost | ridge, random_forest, catboost |

## b1 대비 RMSE 차이의 95% 구간

음수는 b1보다 오차가 작다는 뜻이다. 구간이 0을 포함하면 개선의 불확실성이 남는다.

| 국면·방식 | 모델 | RMSE 차이 | 95% 구간 |
|---|---|---:|---|
| up1_window | ridge | -2.506 | [-3.476, -1.666] |
| up1_window | random_forest | -3.274 | [-4.410, -2.041] |
| up1_window | catboost | -2.885 | [-3.608, -2.203] |
| up1_slope | ridge | -2.484 | [-3.675, -1.607] |
| up1_slope | random_forest | -3.294 | [-4.848, -1.896] |
| up1_slope | catboost | -3.101 | [-3.774, -2.521] |
| corr_window | ridge | -0.533 | [-0.899, -0.214] |
| corr_window | random_forest | -0.475 | [-0.856, -0.127] |
| corr_window | catboost | -0.532 | [-0.831, -0.290] |
| corr_slope | ridge | -0.314 | [-0.591, -0.074] |
| corr_slope | random_forest | -0.272 | [-0.529, -0.034] |
| corr_slope | catboost | -0.311 | [-0.490, -0.153] |
| up2_window | ridge | -3.808 | [-4.924, -2.537] |
| up2_window | random_forest | -7.214 | [-8.694, -5.592] |
| up2_window | catboost | -7.666 | [-9.050, -6.239] |
| up2_slope | ridge | -3.159 | [-3.873, -2.421] |
| up2_slope | random_forest | -6.417 | [-7.693, -5.272] |
| up2_slope | catboost | -6.529 | [-7.663, -5.474] |
| down_window | ridge | -1.332 | [-1.996, -0.672] |
| down_window | random_forest | -3.402 | [-4.381, -2.561] |
| down_window | catboost | -3.423 | [-4.282, -2.737] |
| down_slope | ridge | -1.085 | [-1.696, -0.442] |
| down_slope | random_forest | -2.697 | [-3.813, -1.218] |
| down_slope | catboost | -2.793 | [-3.890, -1.288] |
| rec_window | ridge | -1.118 | [-1.500, -0.735] |
| rec_window | random_forest | -1.746 | [-2.201, -1.313] |
| rec_window | catboost | -1.747 | [-2.130, -1.360] |
| rec_slope | ridge | -0.828 | [-1.114, -0.543] |
| rec_slope | random_forest | -1.172 | [-1.542, -0.823] |
| rec_slope | catboost | -1.509 | [-1.831, -1.187] |
| up3_window | ridge | -2.145 | [-2.864, -1.485] |
| up3_window | random_forest | -4.470 | [-5.476, -3.518] |
| up3_window | catboost | -4.587 | [-5.354, -3.828] |
| up3_slope | ridge | -2.123 | [-2.849, -1.393] |
| up3_slope | random_forest | -4.918 | [-5.852, -3.876] |
| up3_slope | catboost | -4.222 | [-4.862, -3.583] |

## 해석 범위

- 이 결과는 고정된 모델의 동네 외부 평가다. 새로운 시기나 새로운 시장 국면에 대한 예측력을 검증한 것은 아니다.
- 기본 15개 특성과 결측 표시만 사용했다. 통제변수 확장 모델·SHAP·특성 유형표는 아직 계산하지 않았다.
- 부트스트랩 구간은 단지를 묶어 1,000회 재표집한 결과이며, 모델 재학습의 불확실성은 포함하지 않는다.
- 모든 MAE·순위상관·상하위 20% 차이와 구간은 `test_metrics_v1.csv`, `test_intervals_v1.csv`에 있다.
- 상수 예측인 b0의 순위상관은 정의되지 않으므로 빈 값이다. 상하위 차이는 동점 처리 후 0이다.
- 큰 오차를 보고 test 표본을 제거하거나 모델을 다시 튜닝하지 않았다.
