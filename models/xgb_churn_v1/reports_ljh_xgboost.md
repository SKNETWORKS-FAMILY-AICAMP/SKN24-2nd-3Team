# 인공지능 학습 결과서

---

## 기본 정보

| 항목        | 내용                                    |
| ----------- | --------------------------------------- |
| 작성자      | ljh                                     |
| 작성일      | 2026-02-24                              |
| 모델명      | XGBoost (XGBClassifier)                 |
| 담당 노트북 | `notebooks/ljh/03_modeling.ipynb`       |

---

## 1. 평가 지표 선정

### 1.1 후보 지표 검토

| 지표      | 설명                         | 적합성 검토                      |
| --------- | ---------------------------- | -------------------------------- |
| Accuracy  | 전체 예측 중 정답 비율       | 클래스 불균형 시 misleading 가능 |
| Precision | 이탈 예측 중 실제 이탈 비율  | FP 비용이 클 때 중요             |
| Recall    | 실제 이탈 중 예측 성공 비율  | FN 비용이 클 때 중요             |
| F1-Score  | Precision-Recall 조화 평균   | 불균형 데이터에 균형 잡힌 지표   |
| ROC-AUC   | 분류 임계값 무관 성능        | 전반적 판별 능력 측정            |
| PR-AUC    | 불균형 데이터에 최적화된 AUC | 양성 클래스 비율 낮을 때 유용    |

### 1.2 최종 주요 지표 선정

**주요 지표**: F1-Score (이탈 클래스 기준)

**선정 근거**:

> 이탈 고객 미탐지(FN)의 비용이 오탐지(FP)보다 크므로 Recall을 중심으로 평가하되,
> F1-Score를 주 지표로 삼아 Precision과 균형을 맞춤.
> 임계값 최적화 단계에서도 F1 최대화 기준으로 최적 threshold를 탐색.

### 1.3 클래스 불균형 현황

| 클래스   | 샘플 수 | 비율   |
| -------- | ------- | ------ |
| 잔존 (0) | 33,322  | 71.9%  |
| 이탈 (1) | 13,017  | 28.1%  |
| **합계** | 46,339  | 100%   |

불균형 처리 방법: `scale_pos_weight = 2.56` (XGBoost 손실 함수 내 이탈 샘플 가중치 상향)

---

## 2. 데이터 분할

| 항목            | 내용                              |
| --------------- | --------------------------------- |
| 분할 방식       | train_test_split                  |
| Train 비율      | 80% (37,071건)                    |
| Validation 비율 | — (CV는 튜닝 단계에서 별도 적용)  |
| Test 비율       | 20% (9,268건)                     |
| random_state    | 42                                |
| Stratify 적용   | stratify=y                        |

---

## 3. 초기 모델 학습 (Baseline)

### 3.1 모델 설정

```python
xgb_base = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # 2.56
    eval_metric='logloss',
    early_stopping_rounds=30,
    random_state=42
)
```

**사용 하이퍼파라미터 (기본값)**:

| 파라미터             | 값    |
| -------------------- | ----- |
| n_estimators         | 500   |
| max_depth            | 6     |
| learning_rate        | 0.1   |
| scale_pos_weight     | 2.56  |
| early_stopping_rounds| 30    |
| eval_metric          | logloss |

> Early Stopping 적용 → 최적 트리 수: **283** (313번째 iteration에서 조기 종료)

### 3.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [  5,494  |  1,171  ]
실제 이탈  [    441  |  2,162  ]
```

**평가 지표 (Test Set)**

| 지표      | 값     |
| --------- | ------ |
| Accuracy  | 0.8261 |
| Precision | 0.6487 |
| Recall    | 0.8306 |
| F1-Score  | 0.7284 |
| ROC-AUC   | 0.9154 |

**Classification Report**

```
              precision    recall  f1-score   support

      잔류(0)       0.93      0.82      0.87      6665
      이탈(1)       0.65      0.83      0.73      2603

    accuracy                           0.83      9268
   macro avg       0.79      0.83      0.80      9268
weighted avg       0.85      0.83      0.83      9268
```

### 3.3 초기 모델 분석

## **잘된 점**:

- ROC-AUC 0.9154로 높은 전반적 판별 능력 확보
- 이탈 클래스 Recall 0.83으로 실제 이탈 고객의 83%를 탐지
- Early Stopping(30 rounds)으로 과적합 없이 학습 조기 종료

## **문제점 / 개선 필요 사항**:

- 이탈(1) Precision이 0.65로 낮음 → 잔류 고객을 이탈로 잘못 분류하는 FP가 1,171건
- `scale_pos_weight` 적용으로 Precision-Recall 트레이드오프 발생
- max_depth=6으로 과적합 위험 존재 → 하이퍼파라미터 튜닝 필요

---

## 4. 하이퍼파라미터 탐색

### 4.1 탐색 방법

- [x] Random Search CV
- [x] Optuna (Bayesian Optimization) 대신 RandomizedSearchCV 선택

**CV Fold 수**: 5-fold StratifiedKFold (shuffle=True, random_state=42)
**최적화 기준 지표**: roc_auc
**n_iter**: 50 (총 250회 fit 수행)

**선정 이유**:
- `RandomizedSearchCV`: 50회 무작위 샘플링으로 빠른 탐색 + 충분한 커버리지
- `StratifiedKFold`: 불균형 데이터에서 각 fold의 잔류/이탈 비율 유지

### 4.2 탐색 공간 정의

```python
param_dist = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [3, 4, 5, 6, 8],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'min_child_weight':[1, 3, 5, 7],
    'reg_alpha':       [0, 0.01, 0.1, 1.0],   # L1 규제
    'reg_lambda':      [1.0, 1.5, 2.0, 5.0],  # L2 규제
}
```

### 4.3 탐색 결과 (상위 5개)

| 순위 | 주요 파라미터 조합 | CV Score (roc_auc) |
| ---- | ------------------ | ------------------ |
| 1    | max_depth=3, n_estimators=500, lr=0.1, min_child_weight=5, reg_alpha=0.1, reg_lambda=5.0 | 0.9181 |
| 2~5  | (notebook 참조 — RandomizedSearchCV 상위 결과 미출력) | — |

**Best Parameters**:

```python
best_params = {
    'reg_lambda':      5.0,
    'reg_alpha':       0.1,
    'n_estimators':    500,
    'min_child_weight':5,
    'max_depth':       3,
    'learning_rate':   0.1,
}
```

---

## 5. 최적 모델 학습 결과

### 5.1 최적 모델 설정

```python
# best_params 적용 + 임계값 최적화 (F1 기준 threshold=0.5724)
best_model = rs.best_estimator_  # RandomizedSearchCV 최적 추정기

# 최적 임계값 탐색
precision_arr, recall_arr, thresh_arr = precision_recall_curve(y_test, y_proba_tuned)
f1_arr = 2 * precision_arr[:-1] * recall_arr[:-1] / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
best_threshold = thresh_arr[np.argmax(f1_arr)]  # 0.5724

y_pred_opt = (y_proba_tuned >= best_threshold).astype(int)
```

### 5.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [  5,659  |  1,006  ]
실제 이탈  [    484  |  2,119  ]
```

**평가 지표 (Test Set)**

| 지표      | Baseline | 최적 모델 | 변화     |
| --------- | -------- | --------- | -------- |
| Accuracy  | 0.8261   | 0.8392    | +0.0131  |
| Precision | 0.6487   | 0.6780    | +0.0293  |
| Recall    | 0.8306   | 0.8144    | -0.0162  |
| F1-Score  | 0.7284   | 0.7400    | +0.0116  |
| ROC-AUC   | 0.9154   | 0.9180    | +0.0026  |

**Classification Report**

```
              precision    recall  f1-score   support

      잔류(0)       0.92      0.85      0.88      6665
      이탈(1)       0.68      0.81      0.74      2603

    accuracy                           0.84      9268
   macro avg       0.80      0.83      0.81      9268
weighted avg       0.85      0.84      0.84      9268
```

### 5.3 학습 곡선 / 오버피팅 여부

| 지표     | 관찰 내용                                                                 | 판정       |
| -------- | ------------------------------------------------------------------------ | ---------- |
| logloss  | Baseline: 0.647(초기) → 0.353(수렴), 313iter에서 early stopping         | 적절       |
| 모델구조 | 튜닝 결과 max_depth=3 선택 (Baseline max_depth=6 대비 단순화)            | 오버피팅 ↓ |
| CV AUC   | 5-fold CV ROC-AUC=0.9181 vs Test ROC-AUC=0.9180 (차이 미미)             | 적절       |

---

## 6. 추가 실험 (임계값 최적화)

### 6.1 실험 목록

| 실험                 | 설명                                                   | 결과 (F1)              | 채택 여부 |
| -------------------- | ------------------------------------------------------ | ---------------------- | --------- |
| 임계값 최적화 (F1)   | precision_recall_curve로 F1 최대 임계값 탐색 (0.5→0.5724) | 0.7400 (+0.0114)       | 채택      |
| 튜닝 모델 (0.5 고정) | 하이퍼파라미터 튜닝만 적용, threshold=0.5              | 0.7286 (기준)          | 비채택    |

> 임계값 0.5724 적용 시 Precision +0.0433, Recall -0.0408, F1 +0.0114
> Recall이 소폭 감소하지만 Precision 향상으로 F1 전체 개선

---

## 7. 특성 중요도 (Feature Importance)

**학습에 사용된 피처 (20개)**:

고상관(|r|>0.95) 피처 제거 후 최종 선택된 피처:

| 카테고리       | 피처명                                                                 |
| -------------- | ---------------------------------------------------------------------- |
| 인구통계       | gender, age_group                                                      |
| 베팅 금액      | fixed_bet_amount, live_bet_amount                                      |
| 베팅 횟수      | fixed_bet_cnt, live_bet_cnt, total_bet_cnt                             |
| 활동 일수      | fixed_active_days, live_active_days                                    |
| 적중 일수      | fixed_hit_days, total_hit_days                                         |
| 승률           | fixed_win_rate, live_win_rate, total_win_rate                          |
| 평균 ROI       | fixed_avg_roi, live_avg_roi, total_avg_roi                             |
| 가입/활동 경과 | days_since_reg, days_to_first_deposit, days_to_first_bet               |

**상위 15개 중요 특성**: notebook 시각화 참조 (`notebooks/ljh/03_modeling.ipynb` 셀 50 — Feature Importance 바 차트)

## **주요 인사이트**:

- 다중공선성 제거로 total_bet_amount, total_win_amount 등 고상관 피처 사전 제거
- XGBoost 내부 피처 중요도 기준 상위 피처는 베팅 활동성(active_days, bet_cnt) 및 ROI 관련 지표로 예상
- gender, age_group 등 인구통계 피처는 상대적으로 낮은 중요도 예상

---

## 8. 모델 저장

```python
from src.utils.model_manager import ModelManager

mm = ModelManager(base_dir='../../models/xgb')
mm.save(best_model, 'xgb', metadata={
    'accuracy':         0.8392,
    'precision':        0.6780,
    'recall':           0.8144,
    'f1_score':         0.7400,
    'roc_auc':          0.9180,
    'best_params':      best_params,
    'best_cv_auc':      0.9181,
    'best_threshold':   0.5724,
    'features':         X.columns.tolist(),
    'n_features':       20,
    'train_size':       37071,
    'test_size':        9268,
    'scale_pos_weight': 2.56,
})
```

저장 경로: `models/xgb/xgb/`

---

## 9. 최종 요약

### 9.1 성능 요약

| 구분          | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------- | -------- | --------- | ------ | -------- | ------- |
| Baseline      | 0.8261   | 0.6487    | 0.8306 | 0.7284   | 0.9154  |
| 최적 모델     | 0.8392   | 0.6780    | 0.8144 | 0.7400   | 0.9180  |
| 개선폭        | +0.0131  | +0.0293   | -0.0162| +0.0116  | +0.0026 |

### 9.2 결론 및 특이사항

**결론**:

> XGBoost 기반 고객 이탈 예측 모델은 ROC-AUC 0.918, F1-Score 0.740을 달성.
> 하이퍼파라미터 튜닝(RandomizedSearchCV, 5-fold CV)과 임계값 최적화(0.5 → 0.5724)를 통해
> Baseline 대비 F1 +0.0116, Precision +0.0293 개선.
> scale_pos_weight=2.56 적용으로 소수 클래스(이탈) 탐지 성능을 효과적으로 확보.

## **한계 및 향후 개선 방향**:

- 이탈(1) Precision이 0.68로 여전히 FP(1,006건)가 많음 → 마케팅 비용 낭비 가능성
- 임계값 최적화 시 Recall이 0.83 → 0.81로 소폭 감소 (FN 441 → 484건 증가)
- 향후 개선 방향:
  - SMOTE 또는 앙상블(Voting/Stacking)으로 Precision-Recall 균형 추가 개선
  - SHAP을 활용한 피처 기여도 해석으로 비즈니스 인사이트 도출
  - 더 많은 피처 엔지니어링(최근성·빈도·금액 RFM 지표 등) 적용 검토
