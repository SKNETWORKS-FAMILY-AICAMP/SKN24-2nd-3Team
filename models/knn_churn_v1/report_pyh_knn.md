# 인공지능 학습 결과서

## 기본 정보

| 항목        | 내용                                              |
| ----------- | ------------------------------------------------- |
| 작성자      | (박영훈/https://github.com/aprkaos56)                                  |
| 작성일      | 2026-02-24                                        |
| 모델명      | (KNN) |
| 담당 노트북 | `notebooks/pyh/03_modeling.ipynb`          |

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

**주요 지표**: Recall (보조 지표: F1-Score)
**선정 근거**: 고객 이탈 예측에서는 실제 이탈 고객을 놓치는 경우(FN)의 비용이 크다고 판단하여 Recall을 주요 지표로 선정하였으며, 하이퍼 파라미터 탐색 시, `GridSearchCV(scoring='recall')`를 사용하였습니다. 다만, Recall만으로는 오탐(FP) 증가를 반영하기 어려워 Precision, F1-Score를 함께 확인하였습니다.

### 1.3 클래스 불균형 현황

| 클래스   | 샘플 수 | 비율 |
| -------- | ------- | ---- |
| 잔존 (0) |  6,665       |  71.9%    |
| 이탈 (1) |  2,603       |  28.1%    |
| **합계** |  9,268       | 100% |

불균형 처리 방법: (X, y 데이터 분리 과정에서 stratify=y 적용)

---

## 2. 데이터 분할

| 항목            | 내용                                     |
| --------------- | ---------------------------------------- |
| 분할 방식       | train_test_split |
| Train 비율      | 80%                                        |
| Validation 비율 | GridSearchCV, CV=5 사용                                        |
| Test 비율       | 20%                                        |
| random_state    | 42                                       |
| Stratify 적용   | stratify=y 적용                         |

---

## 3. 초기 모델 학습 (Baseline)

### 3.1 모델 설정

```python
# X, y 분리
X = df.drop(columns=['user_id', 'churn', 'country_id'])
y = df['churn']

# 날짜 컬럼(문자열) 제거
X = X.drop(columns=['reg_date', 'first_deposit', 'first_bet'])

# X, y 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 결측치 처리
train_median = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)

# KNN 모델 사용
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)

# KNN 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
kn.fit(X_train_scaler, y_train)

# 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
kn = KNeighborsClassifier()

params = {'n_neighbors': range(1, 15, 2)}
grid = GridSearchCV(kn, params, scoring='recall', cv=5)
grid.fit(X_train_scaler, y_train)
y_pred = grid.best_estimator_.predict(X_test_scaler)

print('최적의 하이퍼 파라미터:', grid.best_params_)
print('최고 CV 점수(recall):', grid.best_score_)
print('학습 데이터 점수:', grid.best_estimator_.score(X_train_scaler,y_train))
print('평가 데이터 점수:', grid.best_estimator_.score(X_test_scaler,y_test))
print(classification_report(y_test, y_pred))
```

**사용 하이퍼파라미터 (기본값)**:

| 파라미터     | 값       |
| ------------ | -------- |
| n_neighbors | 5 |
| weights | 기본값 |
| metric | 기본값 |
| p | 기본값 |

### 3.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [   5813    |    852   ]
실제 이탈  [   1208    |    1395   ]
```

**평가 지표 (Test Set)**

| 지표      | 값  |
| --------- | --- |
| Accuracy  |  0.7777   |
| Precision |  0.6208   |
| Recall    |  0.5359   |
| F1-Score  |  0.5753   |
| ROC-AUC   |  0.7993   |

**Classification Report**

```
              precision    recall  f1-score   support
           0    0.83        0.87     0.85       6665
           1    0.62        0.54     0.58       2603
    accuracy                         0.78       9268
   macro avg    0.72        0.70     0.71       9268
weighted avg    0.77        0.78     0.77       9268
```

### 3.3 초기 모델 분석

## **잘된 점**:
- KNN 모델에 대해 스케일링, 하이퍼파라미터 탐색, 최종 평가까지 전체 흐름을 구현할 수 있었습니다.
- 테스트셋 기준 Accuracy 0.7777, ROC-AUC 0.7993으로 기본적인 분류 성능을 확보할 수 있었습니다.
- Confusion Matrix와 Classification Report를 함께 확인하여 클래스별 성능을 구체적으로 해석할 수 있었습니다.

## **문제점 / 개선 필요 사항**: 
- 이탈 클래스(1) Recall 값이 0.5359로, 실제 이탈 고객의 약 46.4%는 여전히 놓치고 있다는 문제점을 발견하였습니다.
- Precision(0.6208)과 Recall(0.5359) 간 균형 개선의 여지가 있음을 파악할 수 있었습니다
- KNN 모델은 결측치 처리에 민감하므로 추가 전처리 개선점이 필요하다는 점을 알 수 있었습니다.

---

## 4. 하이퍼파라미터 탐색

### 4.1 탐색 방법

- [x] Grid Search CV
- [ ] Random Search CV
- [ ] Optuna (Bayesian Optimization)
- [ ] 수동 조정

**CV Fold 수**: 5-fold CV
**최적화 기준 지표**: recall

### 4.2 탐색 공간 정의

```python
params = {'n_neighbors': range(1, 15, 2)}
```

### 4.3 탐색 결과 (상위 5개)

| 순위 | 주요 파라미터 조합 | CV Score (주요 지표) |
| ---- | ------------------ | -------------------- |
| 1    |   `{'n_neighbors': 5}`                 |      0.5313                |
| 2    |   `{'n_neighbors': 3}`                 |      0.5267                |
| 3    |   `{'n_neighbors': 1}`                 |      0.5237                |
| 4    |   `{'n_neighbors': 7}`                 |      0.5213                |
| 5    |   `{'n_neighbors': 9}`                 |      0.5161                |

**Best Parameters**: 

```python
best_params = {
    'n_neighbors': 5,
}
```

---

## 5. 최적 모델 학습 결과

### 5.1 최적 모델 설정

```python
# 최적 하이퍼파라미터 적용 코드
model = KNeighborsClassifier(**best_params)
model.fit(X_train_scaler, y_train)
```

### 5.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [   5813    |    852   ]
실제 이탈  [   1208    |    1395   ]
```

**평가 지표 (Test Set)**

| 지표      | Baseline | 최적 모델 | 변화 |
| --------- | -------- | --------- | ---- |
| Accuracy  |   0.7777       |   0.7777        |   0   |
| Precision |   0.6208       |   0.6208        |   0   |
| Recall    |   0.5359       |   0.5359        |   0   |
| F1-Score  |   0.5753       |   0.5753        |   0   |
| ROC-AUC   |   0.7993       |   0.7993        |   0   |

**Classification Report**

```
              precision    recall  f1-score   support
           0    0.83        0.87     0.85       6665
           1    0.62        0.54     0.58       2603
    accuracy                         0.78       9268
   macro avg    0.72        0.70     0.71       9268
weighted avg    0.77        0.78     0.77       9268
```

### 5.3 학습 곡선 / 오버피팅 여부

| 지표           | Train | Validation | 판정                 |
| -------------- | ----- | ---------- | -------------------- |
| F1-Score       |  0.6949     |   0.5753         | 약한 오버피팅 가능성 |
| Loss (해당 시) |  해당 없음     |   해당 없음         |     해당 없음                 |

---

## 7. 특성 중요도 (Feature Importance)

> KNN은 거리 기반 모델로, 트리 기반 모델처럼 직접적인 Feature Importance를 제공하지 않음.

**상위 10개 중요 특성**: 해당 없음 (KNN 특성상 미제공)


## **주요 인사이트**: 
KNN 모델의 성능은 하이퍼파라미터(n_neighbors)보다 **전처리(결측치 처리, 스케일링, 컬럼 선택)**에 더 민감하게 반응하다는 점을 알 수 있었습니다. 또한, 이탈 예측에서 Recall 값의 개선을 위해서는 KNN 추가 튜닝뿐 아니라, 다른 모델(예: Logistic Regression, RandomForest, XGBoost)과의 비교가 필요하다는 점도 파악할 수 있었습니다.

---

## 8. 모델 저장

```python
from src.utils.model_manager import ModelManager
mm = ModelManager(base_dir='models')
mm.save(model, 'pyh_knn_v1', metadata={
    'accuracy': 0.7777,
    'f1_score': 0.5753,
    'roc_auc': 0.7993,
    'best_params': {'n_neighbors': 5},
    'data': 'data/processed/ljh_preprocessed.csv',
    'random_state': 42,
})
```

저장 경로: `models/pyh_knn_v1/`

---

## 9. 최종 요약

### 9.1 성능 요약

| 구분      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------- | -------- | --------- | ------ | -------- | ------- |
| Baseline  |  0.7777        |   0.6208        |   0.5359     |   0.5753       |  0.7993       |
| 최적 모델 |   0.7777       |    0.6208       |    0.5359    |    0.5753      |   0.7993      |
| 개선폭    |    0.0000      |     0.0000      |    0.0000    |    0.0000      |    0.0000     |

### 9.2 결론 및 특이사항

**결론**:

> KNN 모델을 활용하여 고객 이탈 예측 모델을 구축하고, 결측치 처리(train median), 표준화(StandardScaler), GridSearchCV(scoring='recall', cv=5)를 적용하여 하이퍼파라미터 탐색을 수행하였습니다.
n_neighbors를 1~13(홀수) 범위에서 탐색한 결과 최적 파라미터 값은 5로 도출되었으며, 이는 KNN 기본값과 동일하였습니다.
최종 모델의 테스트셋 성능은 Accuracy 0.7777, Precision 0.6208, Recall 0.5359, F1-Score 0.5753, ROC-AUC 0.7993으로 확인되었습니다.
Baseline과 최적 모델의 성능이 동일하게 나타난 것은, 현재 실험 조건에서 n_neighbors 단일 파라미터 조정보다 **전처리 과정(결측치 처리, 스케일링, 피처 구성)**의 영향이 더 크다는 점을 알 수 있었습니다.

## **한계 및 향후 개선 방향**: 
KNN 모델은 거리 기반 모델 특성상 전처리에 매우 민감하므로, 결측치 처리 방식에 좀 더 신경쓸 필요가 있다고 생각합니다.
KNN 모델 외에도 Logistic Regression, RandomForest, XGBoost 등과 같은 모델과의 성능 비교를 통해 더욱 적합한 모델을 선정할 필요가 있다고 생각합니다.

---

_본 결과서는 SKN24 2차 프로젝트 — 인터넷 베팅 플랫폼 고객 이탈 예측 (A4 House 팀) 을 위해 작성되었습니다._