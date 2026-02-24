# 인공지능 학습 결과서

> **작성 가이드**: 본 양식을 복사하여 `models/{model_name}/report_{initials}_{model_name}.md` 로 저장 후 작성하세요.
> 예: `models/xgb_churn_v1/report_ljh_xgboost.md`

---

## 기본 정보

| 항목        | 내용                                                |
| ----------- | ------------------------------------------------- |
| 작성자      | 박수영/git/suyoung6279@gmail.com                    |
| 작성일      | 2026-02-23                                         |
| 모델명      |  Random Forest,                                    |
| 담당 노트북 | `notebooks/psy/03_modeling.ipynb`                   |

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

**주요 지표**: **Reacall**
**선정 근거**: 본 분석의 목적은 이탈 가능성이 높은 유저를 예측해, 그들의 이탈을 방지하는 해결책을 마련합니다.
이탈 고객을 미탐지 했을 때 발생하는 고객 상실 비용이, 유지 고객을 이탈로 오탐지하여 발생하는 마케팅 비용보다 압도적으로 크기 때문에 Recall을 최우선 지표로 선정하였습니다.


### 1.3 클래스 불균형 현황

| 클래스   | 샘플 수 | 비율 |
| -------- | ------- | ---- |
| 잔존 (0) |  33322  | 72%  |
| 이탈 (1) |  13017  | 28%  |
| **합계** |  46339  | 100% |

불균형 처리 방법: 없음

---

## 2. 데이터 분할

| 항목            | 내용                                     |
| --------------- | ---------------------------------------- |
| 분할 방식        |      train_test_split                   |
| Train 비율      | 75%                                      |
| Validation 비율 | %                                        |
| Test 비율       | 25%                                      |
| random_state    | 42                                      |
| Stratify 적용   | stratify=y                               |

---

## 3. 초기 모델 학습 (Baseline)

### 3.1 모델 설정

```python
# 초기 모델 코드 붙여넣기
rf_clf = RandomForestClassifier(max_depth=7, random_state=42)
```

**사용 하이퍼파라미터 (기본값)**:

| 파라미터     | 값       |
| ------------ | -------- |
| max_depth    | 7 |

### 3.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [  7629   |   702   ]
실제 이탈  [  1512   |   1733  ]
```

**평가 지표 (Test Set)**

| 지표      | 값  |
| --------- | --- |
| Accuracy  |  0.8081   |
| Precision |  0.7117   |
| Recall    |  0.5325   |
| F1-Score  |  0.5325   |
| ROC-AUC   |  0.7241   |

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.83      0.92      0.87      8331
           1       0.71      0.53      0.61      3254

    accuracy                           0.81     11585
   macro avg       0.77      0.72      0.74     11585
weighted avg       0.80      0.81      0.80     11585
```

### 3.3 초기 모델 분석

## **잘된 점**: Accuracy가 **80.8%**로, 전반적인 예측 성능이 안정적이다.

## **문제점 / 개선 필요 사항**: : 핵심 목표인 이탈 고객 탐지율(Recall)이 **53.3%**에 머물러 있다.

---

## 4. 하이퍼파라미터 탐색

### 4.1 탐색 방법

- [ ] Grid Search CV
- [ ] Random Search CV
- [x] hyperopt 
- [ ] Optuna (Bayesian Optimization)
- [x] 수동 조정

**CV Fold 수**: cv=5
**최적화 기준 지표**: recall

### 4.2 탐색 공간 정의

```python
search_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
    'max_depth': hp.quniform('max_depth', 3, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
}

def objective(space):
    model = RandomForestClassifier(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        min_samples_split=int(space['min_samples_split']),
        min_samples_leaf=int(space['min_samples_leaf']),
        n_jobs=-1,
        random_state=42
    )
```

### 4.3 탐색 결과 (상위 5개)

| 순위 | 주요 파라미터 조합             | CV Score (주요 지표) |
| ---- | ------------------         | -------------------- |
| 1    |   hyperopt                 |   0.5758                   |
| 2    |   optuna                   |   0.5725                   |
| 3    |   RandimizedSearchCV       |   0.5694                   |
| 4    |   GridSearchCV             |   0.5448                   |
| 5    |                            |                      |

**Best Parameters**:

```python
best_params = {
    'n_estimators':350,
    'min_samples_split':12,
    'max_depth':19,
    'min_samples_leaf':1,
    'random_state':42
}

```

---

## 5. 최적 모델 학습 결과

accuracy score : 0.8140699179974105

precision_score : 0.7092846270928462

recall_score : 0.5728334357713584

roc_auc_score : 0.7405638790908166

f1_score : 0.6337980278816729



              precision    recall  f1-score   support

           0       0.84      0.91      0.88      8331
           1       0.71      0.57      0.63      3254

    accuracy                           0.81     11585
   macro avg       0.78      0.74      0.75     11585
weighted avg       0.81      0.81      0.81     11585


### 5.1 최적 모델 설정

```python
# 최적 하이퍼파라미터 적용 코드
best_rf = RandomForestClassifier(
    n_estimators=int(best_val['n_estimators']),
    max_depth=int(best_val['max_depth']),
    min_samples_split=int(best_val['min_samples_split']),
    min_samples_leaf=int(best_val['min_samples_leaf']),
    n_jobs=-1,
    random_state=42
)
```

### 5.2 학습 결과

**Confusion Matrix**

```
              예측 잔존  예측 이탈
실제 잔존  [ 7567    |   764   ]
실제 이탈  [ 1390    |  1864   ]
```



**평가 지표 (Test Set)**

| 지표      | Baseline | 최적 모델 | 변화 |
| --------- | -------- | --------- | ---- |
| Accuracy  |  0.8081        |  0.8140         | **+0.0059**    |
| Precision |  0.7117        |  0.7092         | **-0.0025**    |
| Recall    |  0.5325        |  0.5728         | **+0.0403**    |
| F1-Score  |  0.5325        |  0.6337         | **+0.1012**    |
| ROC-AUC   |  0.7241        |  0.7405         | **+0.0164**    |


| Accuracy  |  0.8140   |
| Precision |  0.7092   |
| Recall    |  0.5728   |
| F1-Score  |  0.6337   |
| ROC-AUC   |  0.7405   |

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.84      0.91      0.88      8331
           1       0.71      0.57      0.63      3254

    accuracy                           0.81     11585
   macro avg       0.78      0.74      0.75     11585
weighted avg       0.81      0.81      0.81     11585
```

### 5.3 학습 곡선 / 오버피팅 여부

| 지표            | Train | Validation | 판정                 |
| -------------- | ----- | ---------- | -------------------- |
| F1-Score       | 0.85  |  0.6337    | 과적합                |
| score          | 0.92  |  0.81      | 과적합                |

---

## 6. 추가 실험 


### 6.1 실험 목록

| 실험   | 설명 | 결과 (주요 지표) | 채택 여부 |
| ------ | ---- | ---------------- | --------- |
| 실험 1 |  수동조정  |              |     O      |

과적합 해결을 위해, 복잡도 수동 조정

``` python
rf_clf = RandomForestClassifier(
    n_estimators=500, # 증가
    min_samples_split=30, # 증가
    max_depth=19, 
    min_samples_leaf=1, 
    random_state=42
    )
```
**결과**

훈련 점수 : 0.87 평가 점수 : 0.81

accuracy score : 0.8138

recall_score : 0.5725

f1_score : 0.6333

roc_auc_score : 0.7402

              precision    recall  f1-score   support

           0       0.84      0.91      0.88      8331
           1       0.71      0.57      0.63      3254

    accuracy                           0.81     11585
   macro avg       0.78      0.74      0.75     11585
weighted avg       0.81      0.81      0.81     11585

---

## 7. 특성 중요도 (Feature Importance)

**상위 10개 중요 특성**:

| 순위 | 특성명 | 중요도 |
total_active_days    0.203918
fixed_active_days    0.146626
fixed_bet_cnt        0.095297
total_bet_cnt        0.080843
fixed_bet_amount     0.062941
total_bet_amount     0.050620
fixed_hit_days       0.047549
fixed_win_amount     0.043885
fixed_avg_roi        0.038475
total_win_amount     0.037631

## **주요 인사이트**: 

total/fixed Active Days: 중요도가 가장 높으며(약 35%), 유저가 얼마나 꾸준히 접속하느냐가 이탈 여부를 결정하는 역할을 함

bet Count/Amount: 단순히 베팅 금액이 큰 것보다 얼마나 자주 베팅을 유지하느냐가 이탈 방지에 더 중요한 영향을 줌

hit Days/ROI: 유저가 경기 결과를 맞추는 경우, 이탈을 방지하는 것을 확인함

---

## 8. 모델 저장

```python
from src.utils.model_manager import ModelManager

mm = ModelManager(base_dir='models')
mm.save(
    rf_clf, 
    'psy/psy_RF_Churn_v1',  
    metadata={
        'accuracy': 0.8150,
        'recall': 0.5734,
        'f1_score': 0.6352,
        'roc_auc': 0.7414,
        'best_params': {
            'n_estimators': 500,
            'min_samples_split': 30,
            'max_depth': 21,
            'min_samples_leaf': 2
        },
        'data': '../../data/processed/ljh_preprocessed.csv', 
        'random_state': 42,
        'description': 'tuning to reduce overfitting and improve recall/f1'
    }
)
print("모델 저장이 완료되었습니다: models/JD_RF_Churn_v2/")
```

저장 경로: `models/psy/psy_RF_Churn_v1/`

---

## 9. 최종 요약

### 9.1 성능 요약

| 구분      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------- | -------- | --------- | ------ | -------- | ------- |
| Baseline  | 0.8081  |  0.71     |  0.5325  |  0.5325  |   0.7241  |
| 최적 모델  | 0.8138  |  0.71     |  0.5725  |   0.6333  |   0.7402 |
| 개선폭    | +0.0045   |  -       |  0.040   |   0.1008   |  0.0161  |



### 9.2 결론 및 특이사항

**결론**:

하이퍼파라미터 최적화 및 수동 조정을 통해 초기 모델 대비 Recall을 53.3%에서 57.3%로 약 4%p 향상시켰으며,
F1-Score는 약 10%p 상승하는 것을 확인할 수 있음

과적합 해결 : 
초기 최적 모델에서 나타난 과적합 현상을 하이퍼 파라미터 상향 조정을 통해 개선하였음

>

## **한계 및 향후 개선 방향**:

기대했던 만큼의 높은 재현율(recall)이 나오지 못하여, 향후 임계치 조정 등의 방법으로 개선할 필요성이 있음
---

_본 결과서는 SKN24 2차 프로젝트 — 인터넷 베팅 플랫폼 고객 이탈 예측 (A4 House 팀) 을 위해 작성되었습니다._
