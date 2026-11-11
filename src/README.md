# src

계산은 모두 여기서 한다. 규칙 값은 `config/`, 확정 결과는 `reports/final/`·`data/`에 둔다.
모든 스크립트는 **저장소 최상위에서** `python src/<폴더>/<파일>.py`로 실행한다. 키가 필요한 스크립트는 환경변수에서만 읽는다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `pipeline/` | 실거래 수집·검증, 호실·시점 특성, K-apt 단지 연결(crosswalk v1→v3), 계약 시점 역·학교 거리, 한국어 데이터셋. 모듈끼리 서로 import하므로 한 폴더에 둔다 |
| `geo/` | 카카오 지오코딩(내 PC에서 실행)과 좌표 반영 → v3 |
| `features/` | 시공사·브랜드·토지거래허가구역 |
| `dataset/` | v3 → v4 학교 열 정리 |
| `buildreg/` | 건축물대장 조회(내 PC에서 실행)와 반영 → v5 |
| `regime/` | 자체 지수, 국면 판정 규칙, 부트스트랩, 공식 지수 비교, 파일럿 |
| `cells/` | `build_cells.py`: v5에서 창·기울기 셀×국면 표와 시점 특성·통제변수·제외 내역 생성 |
| `model/`, `evaluate/`, `report/` | 예정: 모델 → 평가·유형 → reports/final |

## 데이터 재생성 순서

원자료(`data/raw/`)가 있어야 한다(`docs/data.md` §2). 중간 산출물은 `data/processed/`에 쌓이고 Git에는 없다.

| 단계 | 스크립트 | 출력 |
|---|---|---|
| 1. 2020~2026 수집·검증 | `pipeline/collect_seoul_sales.py` → `verify_seoul_collection.py` → `finalize_seoul_collection.py` | `data/raw/seoul_sales_2020_2026/`, `reports/seoul_sales_2020_2026/` |
| 2. 호실·시점 특성 | `pipeline/build_unit_features.py` → `build_market_features.py` | `data/processed/` |
| 3. K-apt 단지 연결 | `pipeline/collect_complex.py` → `build_internal_features.py` → `finalize_internal_features.py` → `apply_internal_static_assumption.py` → `improve_complex_crosswalk.py` → `apply_crosswalk_v2.py` → `improve_complex_crosswalk_v3.py` → `apply_crosswalk_v3.py` | `reports/crosswalk_v3/`, `data/processed/` |
| 4. 2017~2019 연장 | `pipeline/extend_2017_2019.py` (같은 규칙으로 새 연도만 판정) | `..._2017_2026_internal_static_assumption_crosswalk_v3.csv` |
| 5. 좌표·역·학교 | `geo/geocode_kakao.py`(내 PC) → `geo/apply_geocode.py` (내부에서 `pipeline/build_access_features_v2.py`, `build_korean_dataset_v2.py` 호출) | `data/final/…_v3.csv` |
| 6. 브랜드·토허 | `features/add_brand_tohuh.py` (5단계 전에 실행해 두면 v3에 합쳐짐) | `data/features/` |
| 7. 학교 열 정리 | `dataset/make_v4_school_columns.py` | `data/final/…_v4.csv` |
| 8. 건축물대장 | `buildreg/fetch_building_register.py`(내 PC) → `buildreg/apply_building_register.py` | `data/final/…_v5.csv`, `data/buildreg/apply_report.json` |
| 9. 압축 | `xz -T0 -6 -k data/final/아파트_실거래_전처리_2017_2026_v5.csv` | `data/final/…_v5.csv.xz` (커밋) |

## 국면

```
python src/regime/build_index_2017_2026.py   # 자체 지수 (data/processed 필요)
python src/regime/bootstrap_regime.py        # 단지 부트스트랩 200회
python src/regime/compare_official.py        # 공식 지수 비교, 확정 국면 → data/regime/official_comparison.json
```

판정 규칙은 `regime/turning_points.py` 한 곳에만 있다. `tests/test_regime.py`가 저장된 지수로 확정 국면이 다시 나오는지 확인한다.

## 셀×국면 표

```
python src/cells/build_cells.py
python tools/verify_cells.py
```

입력은 커밋된 v5·자체지수·시설 이력만 필요하다. 별도 API 키나 `data/processed`는 필요 없다.
출력과 표본 변화는 `reports/final/cells/README.md`에 설명했다. 모델 학습은 하지 않는다.
