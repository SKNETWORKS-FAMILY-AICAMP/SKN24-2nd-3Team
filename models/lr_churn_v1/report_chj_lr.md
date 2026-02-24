# 인공지능 학습 결과서

> **작성 가이드**: 본 양식을 복사하여 `models/{model_name}/report_{initials}_{model_name}.md` 로 저장 후 작성하세요.
> 예: `models/xgb_churn_v1/report_ljh_xgboost.md`

---

## 기본 정보

| 항목        | 내용                                              |
| ----------- | ------------------------------------------------- |
| 작성자      | (최하진 / hun6684)                                  |
| 작성일      | 2026-02-24                                       |
| 모델명      | Logistic Regression |
| 담당 노트북 | `notebooks/{chj}/03_modeling.ipynb`          |

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

**주요 지표**: (F1-score)
**선정 근거**: 모델링 결과 정밀도와 재현율이 균형을 이루는 지점을 찾기 위해, 두 지표의 조화 평균인 F1-Score를 핵심 성능 지표로 선정하였습니다.

### 1.3 클래스 불균형 현황

| 클래스   | 샘플 수 | 비율 |
| -------- | ------- | ---- |
| 잔존 (0) | 33321 | 71.91% |
| 이탈 (1) | 13016 | 28.09% |
| **합계** | 46337 | 100% |

불균형 처리 방법: 없음
---

## 2. 데이터 분할

| 항목            | 내용                                     |
| --------------- | ---------------------------------------- |
| 분할 방식       | train_test_split  |
| Train 비율      | 80%                                        |
| Test 비율       | 20%                                        |
| random_state    | 42                                       |
| Stratify 적용   | stratify=y                      |

---

## 3. 초기 모델 학습 (Baseline)

### 3.1 모델 설정

```python
# 초기 모델 코드 붙여넣기
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
```

**사용 하이퍼파라미터 (기본값)**:

| 파라미터     | 값       |
| ------------ | -------- |
| (파라미터명) | (기본값) |

### 3.2 학습 결과

**Confusion Matrix**

```
            예측 잔존  예측 이탈
실제 잔존  [ TN 5773 | FP  892 ]
실제 이탈  [ FN 999  | TP 1694 ]
```

**평가 지표 (Test Set)**

| 지표      | 값  |
| --------- | --- |
| Accuracy  | 0.80 |
| Precision | 0.66 |
| Recall    | 0.65 |
| F1-Score  | 0.65 |
| ROC-AUC   | 0.86 |

**Classification Report**

```
              precision    recall  f1-score   support
           0    0.86        0.87    0.87        6665
           1    0.66        0.65    0.65        2603
    accuracy                        0.81        2603
   macro avg    0.76        0.76    0.76        9268    
weighted avg    0.81        0.81    0.81        9268
```

### 3.3 초기 모델 분석

## **잘된 점**:
하이퍼파라미터 튜닝 전임에도 불구하고 ROC-AUC 0.86, F1-Score 0.65라는 수치를 기록했습니다. 이는 데이터 전처리와 스케일링 단계가 모델 학습에 적합하게 수행되었습니다.

## **문제점 / 개선 필요 사항**:
특성 중요도 분석 결과, 특정 변수(total_active_days)의 영향력이 다른 변수들에 비해 압도적으로 높게 나타나 개선사항이 필요합니다.

---

## 4. 하이퍼파라미터 탐색

### 4.1 탐색 방법

- [x] Grid Search CV
- [ ] Random Search CV
- [ ] Optuna (Bayesian Optimization)
- [ ] 수동 조정

**CV Fold 수**: 5-fold StratifiedKFold
**최적화 기준 지표**: f1

### 4.2 탐색 공간 정의

```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs'], 
    'max_iter': [1000]   
}
```

### 4.3 탐색 결과 (상위 5개)

| 순위 | 주요 파라미터 조합 | CV Score (주요 지표:mean_test_score) |
| ---- | ------------------ | -------------------- |
| 1    |  {'C': 100, 'max_iter': 1000, 'solver': 'lbfgs'} | 0.656420 |
| 2    | {'C': 10, 'max_iter': 1000, 'solver': 'lbfgs'} | 0.656417 |
| 3    | {'C': 1, 'max_iter': 1000, 'solver': 'lbfgs'} | 0.653786  |
| 4    | {'C': 0.1, 'max_iter': 1000, 'solver': 'lbfgs'} | 0.648164 |
| 5    |  {'C': 0.01, 'max_iter': 1000, 'solver': 'lbfgs'} | 0.609900 |

**Best Parameters**:

```python
best_params = {
    'C': 100,
    'max_iter': 1000,
    'solver': 'lbfgs'
}
```

---

## 5. 최적 모델 학습 결과

### 5.1 최적 모델 설정

```python
# 최적 하이퍼파라미터 적용 코드
model = GridSearchCV(LogisticRegression(random_state=42), 
                           param_grid, 
                           cv=5, 
                           #scoring='roc_auc', 
                           #scoring='recall',
                           scoring='f1',
                           n_jobs=-1) 
```

### 5.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [ TN  5763 | FP  902 ]
실제 이탈  [ FN   900 | TP 1703 ]
```

**평가 지표 (Test Set)**

| 지표      | Baseline | 최적 모델 | 변화 |
| --------- | -------- | --------- | ---- |
| Accuracy  |  0.8057  |  0.8056   | - 0.0001 |
| Precision |  0.66    |  0.65     |  - 0.1  |
| Recall    |  0.65    |  0.65     |  0  |
| F1-Score  |  0.65    |  0.65     |  0  |
| ROC-AUC   |  0.8636  |  0.8638   | - 0.0001 |

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.86      0.86      0.86      6665
           1       0.65      0.65      0.65      2603

    accuracy                           0.81      9268
   macro avg       0.76      0.76      0.76      9268
weighted avg       0.81      0.81      0.81      9268

```

### 5.3 학습 곡선 / 오버피팅 여부

| 지표           | Train | Validation | 판정                 |
| -------------- | ----- | ---------- | -------------------- |
| F1-Score       | 0.66 |  0.65 | (적절) |

---

## 6. 특성 중요도 (Feature Importance)

> 모델이 특성 중요도를 제공하는 경우 작성 (Random Forest, XGBoost 등).

**중요 특성**:

--- 이탈 확률을 높이는 주요 요인 (Top 5 Positive) ---
              Feature  Coefficient
14     fixed_hit_days     2.599796
15      live_hit_days     1.775588
11  fixed_active_days     1.772165
20      fixed_avg_roi     1.390126
6     live_win_amount     0.963763

--- 이탈 확률을 낮추는 주요 요인 (Top 5 Negative) ---
              Feature  Coefficient
13  total_active_days    -9.521548
3     live_bet_amount    -0.884653
4    total_bet_amount    -0.580711
12   live_active_days    -0.355808
9        live_bet_cnt    -0.285538

## **주요 인사이트**:
핵심 지표: total_active_days

---

## 8. 모델 저장

```python
from src.utils.model_manager import ModelManager
mm = ModelManager(base_dir='models')
mm.save(model, '{initials}_{model_name}_v1', metadata={
    'accuracy': ...,
    'f1_score': ...,
    'roc_auc': ...,
    'best_params': best_params,
    'data': 'data/processed/...',
    'random_state': 42,
})
```

저장 경로: `models/{chj}_{model_name}_v1/`

---

## 9. 최종 요약

### 9.1 성능 요약

| 구분      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------- | -------- | --------- | ------ | -------- | ------- |
| Baseline  | 0.8057 | 0.66 | 0.65 | 0.65 | 0.8636 |
| 최적 모델 | 0.8056 | 0.65 | 0.65 | 0.65 | 0.8638 |

| 지표      | Baseline | 최적 모델 | 변화 |
| --------- | -------- | --------- | ---- |
| Accuracy  |  0.8057  |  0.8056   | - 0.0001 |
| Precision |  0.66    |  0.65     |  - 0.1  |
| Recall    |  0.65    |  0.65     |  0  |
| F1-Score  |  0.65    |  0.65     |  0  |
| ROC-AUC   |  0.8636  |  0.8638   | - 0.0001 |

### 9.2 결론 및 특이사항

**결론**: 하이퍼파라미터 튜닝(C값 조절 등)을 거쳤음에도 성능 향상이 거의 없거나 소폭 하락한 것을 볼 수 있었습니다. 이는 모델 자체의 최적화보다는 결측치 처리와 스케일링 등 앞선 데이터 정제 단계가 모델 성능을 결정짓는 핵심 요인이라는 것을 알 수 잇었습니다. 결과적으로 로지스틱 회귀 모델로 도달할 수 있는 성능의 임계점(Upper Bound)에 도달했다고 판단했습니다.

>

## **한계 및 향후 개선 방향**:
로지스틱 회귀는 변수 간의 선형적인 관계만을 학습하기 때문에, 고객의 행동 패턴 사이에 숨겨진 복잡한 비선형적 상호작용을 모두 포착하기에는 한계가 있었습니다. 현재 모델이 0.86이라는 우수한 ROC-AUC를 기록했으나, 초반의 정확도를 넘어서기 위해서는 XGBoost 등과 같은 트리 기반 앙상블 모델로의 확장이 필수적라고 생각해 향후 개선 방향을 잡았습니다.
---

_본 결과서는 SKN24 2차 프로젝트 — 인터넷 베팅 플랫폼 고객 이탈 예측 (A4 House 팀) 을 위해 작성되었습니다._
