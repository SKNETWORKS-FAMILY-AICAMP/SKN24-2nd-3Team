# SKN24-2nd_3TEAM
# 📊 주제:

## 온라인 베팅 사이트에서의 고객 이탈 예측

---

# 📅 프로젝트 기간

- 2026.02.23(월)~2026.02.24(화)

---

# 1. 팀 소개

## 1-1. 팀명
### 🅰️4 House

## 1-2. 팀원 구성 및 GitHub

| 이름 | 역할 | GitHub |
| --- | --- | --- |
| 박수영 |  | https://github.com/suyoung6279 |
| 박영훈 |  | https://github.com/aprkaos56 |
| 임정희 |  | https://github.com/bigmooon |
| 최하진 |  | https://github.com/hun6684 |

---

# 2. 📋 프로젝트 개요

## 2-1. 프로젝트명

**“인터넷 베팅 플랫폼 고객 이탈 예측 머신러닝 프로젝트”**

<img width="1000" height="533" alt="image" src="https://github.com/user-attachments/assets/ce98c5cc-098c-4177-ac21-db933b68e41c" />
<img width="1000" height="332" alt="image" src="https://github.com/user-attachments/assets/4b25bcd2-a3d7-420c-a6bc-0ae3bc2f8c52" />


**<글로벌 스포츠 베팅 시장 규모, 점유율 및 트렌드 분석 보고서 – 산업 개요 및 2033년까지의 전망>**

출처: [스포츠 베팅 시장의 규모, 트렌드 및 성장 예측 2033](https://www.databridgemarketresearch.com/ko/reports/global-sports-betting-market)


## 2-2. 프로젝트 소개

본 프로젝트는 인터넷 베팅 플랫폼의 사용자 데이터(베팅 빈도, 이용 금액, 적중률, 수익률 등)를 활용하여 **고객 이탈 가능성을 예측하는 머신러닝 모델**을 구축하는 것을 목표로 합니다.

플랫폼 서비스에서는 고객 이탈이 서비스 활성도와 운영 성과에 직접적인 영향을 미치므로, 단순 이용 현황 분석을 넘어 **이탈 가능성을 사전에 예측하고 주요 이탈 요인을 분석하는 데이터 기반 접근**이 중요합니다.

본 프로젝트는 실제 플랫폼 로그 데이터를 바탕으로 전처리, 라벨링, 모델링, 성능 평가, 인사이트 도출까지의 전 과정을 수행하며, 인터넷 베팅 플랫폼 환경에서의 고객 행동 패턴과 이탈의 관계를 정량적으로 분석하고자 합니다.


## 2-3. 🖼️ 프로젝트 필요성

### 1) 플랫폼 서비스 운영에서 고객 이탈은 핵심 지표

인터넷 베팅 플랫폼과 같은 온라인 서비스에서는 사용자 유입뿐 아니라 **지속적인 이용(Retention)**이 서비스 운영 성과에 큰 영향을 미칩니다.

특히 고객 이탈은 이용 빈도 감소, 거래 감소, 서비스 참여도 저하로 이어지기 때문에 사전에 관리할 필요가 있습니다.

### 2) 행동 로그 데이터가 풍부한 도메인 특성

인터넷 베팅 플랫폼은 베팅 빈도, 이용 금액, 적중률, 활동 패턴 등 다양한 사용자 행동 로그가 축적되는 환경입니다.

이러한 데이터는 단순 이용 여부를 넘어 이용 빈도 변화, 거래 금액 변동성, 세션/활동 강도 감소 등과 같은 **이탈 전조 신호**를 반영할 수 있어, 고객 이탈 예측 모델링에 적합한 데이터 구조를 제공합니다.

### 3) 일반적인 온라인 서비스와의 차별성 ← 이게 핵심 Point!!

인터넷 베팅 플랫폼은 일반적인 온라인 서비스에 비해 **반복 이용 주기가 짧고 행동 데이터가 고빈도로 누적**되며, 경기 일정·이벤트·결과에 따라 사용자 활동이 빠르게 변동하는 특성이 있습니다. 이러한 도메인 특성은 사용자별 참여 패턴 변화와 이탈 가능성을 보다 민감하게 관찰할 수 있게 해주므로, **시계열적 행동 변화 분석 및 이탈 조기 탐지** 관점에서 보다 활용 가치가 높습니다.

### 4) 사후 대응보다 사전 예측 중심 분석 필요

실제 운영에서는 고객이 이미 이탈한 뒤 성과 지표를 통해 확인하는 경우가 많습니다. 그러나 이 경우 대응 시점이 늦어질 수 있으며, 이탈 원인 분석도 제한적일 수 있습니다.

따라서 본 프로젝트는 머신러닝 기반 분류 모델을 활용하여 **이탈 가능성을 사전에 탐지**하고, 주요 영향 요인을 도출하는 예측 중심 접근의 가능성을 검토하고자 합니다.


## 2-4. 🎯 프로젝트 목표

### 1) 고객 이탈 기준 정의 및 타깃 변수 설계

- 인터넷 베팅 플랫폼 데이터 특성에 맞게 **고객 이탈(Churn) 기준**을 정의하고,
- 머신러닝 분류 문제에 적합한 타깃 변수(y)를 설계합니다.

### 2) 데이터 전처리 및 피처 엔지니어링

- 결측치/이상치 처리 및 데이터 정제를 수행하고,
- 적중률, 수익률 등 파생 변수를 생성하여 모델 입력 데이터를 구성합니다.

### 3) 머신러닝 기반 이탈 예측 모델 구축 및 성능 비교

- Logistic Regression, Random Forest, XGBoost 등의 분류 모델을 구축하고,
- Precision, Recall, F1-score, ROC-AUC 등 지표를 기반으로 모델 성능을 비교 평가합니다.

### 4) 이탈 영향 요인 분석 및 해석

- Feature Importance 및 모델 해석 결과를 바탕으로 인터넷 베팅 플랫폼 고객 이탈에 영향을 미치는 주요 행동 요인을 도출합니다.

### 5) 운영 관점 시사점 도출

- 예측 결과를 통해 고객 이탈 조기 탐지 가능성을 검토하고, 서비스 운영 및 고객 경험 개선 관점에서 활용 가능한 인사이트를 제시합니다.

---

# 3. 🧰 기술 스택

- Python
    
    [](https://img.shields.io/badge/python-blue?style=for-the-badge&logo=python&logoColor=white)
    
- selenium
    
    [](https://img.shields.io/badge/selenium-green?style=for-the-badge&logo=selenium&logoColor=white)
    
- pandas
    
    [](https://img.shields.io/badge/pandas-yellow?style=for-the-badge&logo=pandas&logoColor=white)
    
- numpy
    
    [](https://img.shields.io/badge/numpy-lightblue?style=for-the-badge&logo=numpy&logoColor=white)
    
- matplotlib
    
    [](https://img.shields.io/badge/matplotlib-black?style=for-the-badge&logo=matplotlib&logoColor=white)
    
- seaborn
    
    [](https://img.shields.io/badge/seaborn-darkblue?style=for-the-badge&logo=seaborn&logoColor=white)
    
- skikit-learn
    
    [](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
    

| Language | Python |
| --- | --- |
| Web Crawling | selenium |
| Data Processing | pandas, numpy, scikit-learn |
| Data Visualization | matplotlib, seaborn |

---

# 4. WBS 및 폴더 구조*(작성 예시)*

```markdown
burger-crime-eda/
├── features/
│   ├── burger/
│   │   ├── lotteria/
│   │   │   ├── crawl_lotteria.py
│   │   │   ├── transform_lotteria.py
│   │   │   └── lotteria.json
│   │   │
│   │   ├── burgerking/
│   │   │   ├── crawl_burgerking.py
│   │   │   ├── transform_burgerking.py
│   │   │   └── burgerking.json
│   │   │
│   │   ├── mcdonalds/
│   │   │   ├── crawl_mcdonalds.py
│   │   │   ├── transform_mcdonalds.py
│   │   │   └── mcdonalds.json
│   │   │
│   │   └── kfc/
│   │       ├── crawl_kfc.py
│   │       ├── transform_kfc.py
│   │       └── kfc.json
│   │
│   ├── population/
│   │   ├── crawl_population.py
│   │   │── transform_population.py
│   │   └── population.json
│   │
│   └── crime/
│       ├── crawl_crime.py
│       │── transform_crime.py
│       └── crime.json
│
├── docs/
│   └── git_strategy.md
├── main.py
├── data.json                    # 최종 통합 파일
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 5. ✅ 데이터 수집 및 출처

## 5-1. 인터넷 베팅 사이트 데이터

- 출처: [The Transparency Project - Data Download](http://www.thetransparencyproject.org/download_index.php)
- 수집 항목
    - 2005년 2월 1일부터 2006년 8월 31일까지의 온라인 스포츠 베팅 기록

## 5-2. 고객 이탈 선정 기준

- 출처: [Help - General Information - Does bwin charge inactivity fees? (UK)](https://help.bwin.com/en/general-information/account/inactive)
- 수집 항목
    - 13개월 연속으로 베팅을 하지 않으면 '비활성'으로 분류

## 5-3. 데이터

- 
- 수집 방식
    - 
    - 

---

# 6. ✅ 주요 지표 정의

## 6-1. 버거지수(Burger Index)

- 정의: 특정 지역 내 주요 버거 프랜차이즈 매장 수의 총합
    
    → 지역의 생활 밀집도 및 상업 환경 수준을 간접적으로 나타내는 대리 변수
    

## 6-2.  지표

- 구성:
- 정의:
- 목적:

## 6-3. 지표

- 구성:
- 정의:
- 목적:
- 

---

# 7. ✅ 데이터 전처리 및 통합

데이터 출처와 수집 방식이 달라 동일한 분석 단위로 결합하기 위해 전처리 과정 수행

### 인구, 범죄 데이터 *(작성 예시)*

- KOSIS 공개 통계(CSV) 활용
- `pandas`, `numpy` 로 DateFrame 생성 및 전처리
- 불필요한 컬럼 제거 및 컬럼(sido, sigungu,population 등) 정리

## 데이터

- 
- 

## 데이터 통합

- 
- 

---

# 8. ℹ️ 분석 인사이트


# 9. 💭 한 줄 회고
### - 박수영: 

### - 박영훈: 

### - 임정희: 

### - 최하: 

