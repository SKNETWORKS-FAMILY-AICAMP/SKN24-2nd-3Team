# ModelManager 사용 설명서

## 개요

`ModelManager`는 sklearn 기반 머신러닝 모델의 **저장, 로드, 조회, 삭제**를 담당하는 유틸리티 클래스입니다. 모델은 `joblib` 포맷으로 직렬화되며, 하이퍼파라미터·평가 지표 등의 메타데이터가 JSON 파일로 함께 관리됩니다.

---

## 초기화

```python
from model_manager import ModelManager

manager = ModelManager(base_dir='models')
```

| 파라미터   | 타입  | 기본값     | 설명                                                    |
| ---------- | ----- | ---------- | ------------------------------------------------------- |
| `base_dir` | `str` | `'models'` | 모델 파일 저장 디렉터리 경로. 존재하지 않으면 자동 생성 |

---

## API

### `save(model, model_name, metadata=None, compress=3)`

학습된 모델을 `.joblib` 파일로 저장하고, 메타데이터를 `_meta.json` 파일로 기록

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

manager.save(
    model,
    model_name='rf_churn_v1',
    metadata={
        'accuracy': 0.895,
        'f1_score': 0.899,
        'roc_auc': 0.948,
        'features': ['tenure', 'monthly_charges', 'contract_type']
    }
)
```

| 파라미터     | 타입           | 설명                                                 |
| ------------ | -------------- | ---------------------------------------------------- |
| `model`      | `Any`          | 학습 완료된 모델 객체                                |
| `model_name` | `str`          | 저장 파일명 (확장자 제외)                            |
| `metadata`   | `dict \| None` | 커스텀 메타데이터 (평가 지표, 피처 목록 등)          |
| `compress`   | `int`          | joblib 압축 레벨 (0~9). 높을수록 파일 작고 저장 느림 |

**반환값**: 저장된 모델 파일의 절대 경로 (`str`)

**생성 파일**:

- `models/rf_churn_v1/rf_churn_v1.joblib` — 직렬화된 모델
- `models/rf_churn_v1/rf_churn_v1_meta.json` — 메타데이터 (아래 구조 참고)

```json
{
  "model_name": "rf_churn_v1",
  "model_type": "RandomForestClassifier",
  "created_at": "2026-02-23T12:00:00.000000",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": null,
    "random_state": 42
  },
  "custom": {
    "accuracy": 0.895,
    "f1_score": 0.899,
    "roc_auc": 0.948,
    "features": ["tenure", "monthly_charges", "contract_type"]
  }
}
```

> `hyperparameters`는 sklearn 모델의 `get_params()`에서 JSON 직렬화 가능한 값만 자동 추출됨

---

### `load(model_name)`

저장된 모델과 메타데이터를 로드

```python
model, meta = manager.load('rf_churn_v1')

# 로드된 모델로 예측
predictions = model.predict(X_test)

# 메타데이터 확인
print(meta['custom']['accuracy'])  # 0.895
```

| 파라미터     | 타입  | 설명                        |
| ------------ | ----- | --------------------------- |
| `model_name` | `str` | 로드할 모델명 (확장자 제외) |

**반환값**: `(model, metadata)` 튜플. 메타데이터 파일이 없으면 `metadata`는 `None`.

## **예외**: 모델 파일이 없으면 `FileNotFoundError` 발생

### `list_models()`

저장 디렉터리 내 모든 모델의 메타데이터를 리스트로 반환

```python
for info in manager.list_models():
    name = info['model_name']
    model_type = info['model_type']
    metrics = info.get('custom', {})
    print(f'{name} ({model_type}) - accuracy: {metrics.get("accuracy", "N/A")}')
```

**반환값**: `list[dict]` — 각 모델의 메타데이터 딕셔너리 목록

---

### `delete(model_name)`

모델 파일(`.joblib`)과 메타데이터(`_meta.json`)를 함께 삭제

```python
manager.delete('rf_churn_v1')
```

---

## 디렉터리 구조

```
models/
├── rf_churn_v1/
│   ├── rf_churn_v1.joblib          # 모델 바이너리
│   ├── rf_churn_v1_meta.json       # 메타데이터
│   └── report_ljh_rf.md            # 학습 결과서 (선택)
├── lr_churn_v1/
│   ├── lr_churn_v1.joblib
│   ├── lr_churn_v1_meta.json
│   └── report_ljh_lr.md
└── gb_churn_v1/
    ├── gb_churn_v1.joblib
    ├── gb_churn_v1_meta.json
    └── report_ljh_gb.md
```

---

## 전체 워크플로우 예시

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model_manager import ModelManager

manager = ModelManager(base_dir='models')

# 여러 모델 학습 → 저장
candidates = {
    'lr_churn_v1': LogisticRegression(max_iter=500),
    'rf_churn_v1': RandomForestClassifier(n_estimators=200),
    'gb_churn_v1': GradientBoostingClassifier(n_estimators=150),
}

for name, model in candidates.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    manager.save(model, name, metadata={
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
    })

# 저장된 모델 비교
for info in manager.list_models():
    m = info.get('custom', {})
    print(f"{info['model_name']:20s} acc={m.get('accuracy')}  f1={m.get('f1_score')}  auc={m.get('roc_auc')}")

# 최적 모델 로드 후 사용
best_model, meta = manager.load('rf_churn_v1')
final_predictions = best_model.predict(X_new)
```
