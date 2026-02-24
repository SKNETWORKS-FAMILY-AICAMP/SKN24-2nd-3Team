# 인공지능 학습 결과서

> **작성 가이드**: 본 양식을 복사하여 `docs/report_{initials}_{model_name}.md` 로 저장 후 작성하세요.
> 예: `docs/report_ljh_xgboost.md`

---

## 기본 정보

| 항목        | 내용                                              |
| ----------- | ------------------------------------------------- |
| 작성자      | (이름/GitHub ID)                                  |
| 작성일      | YYYY-MM-DD                                        |
| 모델명      | (예: XGBoost, Random Forest, Logistic Regression) |
| 담당 노트북 | `notebooks/{initial}/03_modeling.ipynb`           |

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

**주요 지표**: ( )
**선정 근거**:

> (예: 이탈 고객 미탐지(FN)의 비용이 오탐지(FP)보다 크므로 Recall을 중심으로 평가하되,
> F1-Score를 주 지표로 삼아 Precision과 균형을 맞춤)

### 1.3 클래스 불균형 현황

| 클래스   | 샘플 수 | 비율 |
| -------- | ------- | ---- |
| 잔존 (0) |         |      |
| 이탈 (1) |         |      |
| **합계** |         | 100% |

불균형 처리 방법: (없음 / class_weight / SMOTE / 언더샘플링 / 기타)

---

## 2. 데이터 분할

| 항목            | 내용                                     |
| --------------- | ---------------------------------------- |
| 분할 방식       | (예: train_test_split / StratifiedKFold) |
| Train 비율      | %                                        |
| Validation 비율 | %                                        |
| Test 비율       | %                                        |
| random_state    | 42                                       |
| Stratify 적용   | (예: stratify=y)                         |

---

## 3. 초기 모델 학습 (Baseline)

### 3.1 모델 설정

```python
# 초기 모델 코드 붙여넣기
model = ModelName(random_state=42)
```

**사용 하이퍼파라미터 (기본값)**:

| 파라미터     | 값       |
| ------------ | -------- |
| (파라미터명) | (기본값) |

### 3.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [   TN    |    FP   ]
실제 이탈  [   FN    |    TP   ]
```

**평가 지표 (Test Set)**

| 지표      | 값  |
| --------- | --- |
| Accuracy  |     |
| Precision |     |
| Recall    |     |
| F1-Score  |     |
| ROC-AUC   |     |

**Classification Report**

```
              precision    recall  f1-score   support
           0
           1
    accuracy
   macro avg
weighted avg
```

### 3.3 초기 모델 분석

## **잘된 점**:

## **문제점 / 개선 필요 사항**:

---

## 4. 하이퍼파라미터 탐색

### 4.1 탐색 방법

- [ ] Grid Search CV
- [ ] Random Search CV
- [ ] Optuna (Bayesian Optimization)
- [ ] 수동 조정

**CV Fold 수**: (예: 5-fold StratifiedKFold)
**최적화 기준 지표**: (예: f1, roc_auc)

### 4.2 탐색 공간 정의

```python
param_grid = {
    'param_1': [...],
    'param_2': [...],
    # ...
}
```

### 4.3 탐색 결과 (상위 5개)

| 순위 | 주요 파라미터 조합 | CV Score (주요 지표) |
| ---- | ------------------ | -------------------- |
| 1    |                    |                      |
| 2    |                    |                      |
| 3    |                    |                      |
| 4    |                    |                      |
| 5    |                    |                      |

**Best Parameters**:

```python
best_params = {
    'param_1': ...,
    'param_2': ...,
}
```

---

## 5. 최적 모델 학습 결과

### 5.1 최적 모델 설정

```python
# 최적 하이퍼파라미터 적용 코드
model = ModelName(**best_params, random_state=42)
```

### 5.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [   TN    |    FP   ]
실제 이탈  [   FN    |    TP   ]
```

**평가 지표 (Test Set)**

| 지표      | Baseline | 최적 모델 | 변화 |
| --------- | -------- | --------- | ---- |
| Accuracy  |          |           |      |
| Precision |          |           |      |
| Recall    |          |           |      |
| F1-Score  |          |           |      |
| ROC-AUC   |          |           |      |

**Classification Report**

```
              precision    recall  f1-score   support
           0
           1
    accuracy
   macro avg
weighted avg
```

### 5.3 학습 곡선 / 오버피팅 여부

| 지표           | Train | Validation | 판정                 |
| -------------- | ----- | ---------- | -------------------- |
| F1-Score       |       |            | (언더/적절/오버피팅) |
| Loss (해당 시) |       |            |                      |

---

## 6. 추가 실험 (선택)

> 추가로 시도한 기법이 있을 경우 작성. 없으면 섹션 삭제 가능.

### 6.1 실험 목록

| 실험   | 설명 | 결과 (주요 지표) | 채택 여부 |
| ------ | ---- | ---------------- | --------- |
| 실험 1 |      |                  |           |
| 실험 2 |      |                  |           |

---

## 7. 특성 중요도 (Feature Importance)

> 모델이 특성 중요도를 제공하는 경우 작성 (Random Forest, XGBoost 등).

**상위 10개 중요 특성**:

| 순위 | 특성명 | 중요도 |
| ---- | ------ | ------ |
| 1    |        |        |
| 2    |        |        |
| 3    |        |        |
| ...  |        |        |

## **주요 인사이트**:

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

저장 경로: `models/{initials}_{model_name}_v1.joblib`

---

## 9. 최종 요약

### 9.1 성능 요약

| 구분      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------- | -------- | --------- | ------ | -------- | ------- |
| Baseline  |          |           |        |          |         |
| 최적 모델 |          |           |        |          |         |
| 개선폭    |          |           |        |          |         |

### 9.2 결론 및 특이사항

**결론**:

>

## **한계 및 향후 개선 방향**:

---

_본 결과서는 SKN24 2차 프로젝트 — 인터넷 베팅 플랫폼 고객 이탈 예측 (A4 House 팀) 을 위해 작성되었습니다._
