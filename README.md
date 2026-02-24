# SKN24-2nd_3TEAM

---

# 📅 프로젝트 기간

- 2026.02.23(월)~2026.02.24(화)

---

# 1. 팀 소개

## 1-1. 팀명
### 🏠종이의 집에서 수영하기 🏊‍♂️ 

## 1-2. 팀원 구성 및 GitHub

| 사진 | 이름 | GitHub |
| --- | --- | --- |
| <img src="assets/골드베어.png" width="80"> | 박수영 | [suyoung6279](https://github.com/suyoung6279) |
| <img src="assets/연두베오.png" width="80"> | 박영훈 | [aprkaos56](https://github.com/aprkaos56) |
| <img src="assets/보라베어.png" width="80"> | 임정희 | [bigmooon](https://github.com/bigmooon) |
| <img src="assets/아이보리베어.png" width="80"> | 최하진 | [hun6684](https://github.com/hun6684) |

---

# 2. 프로젝트 개요

## 2-1. 프로젝트 소개
본 프로젝트는 약 3만명의 유저와 400만 건 이상의 베팅 로그를 포함한 실측 데이터를 기반으로 스포츠 베팅 플랫폼 `bwin`에서의 고객 행동 패턴과 이탈 간의 상관관계를 분석합니다.

## 2-2. 프로젝트 배경

<img width="1050" alt="image" src="https://github.com/user-attachments/assets/87e04c3a-49ea-4a10-959f-4a4485039585" />
<img width="1050" alt="image" src="https://github.com/user-attachments/assets/45d77237-d7c0-40ed-adc4-0394ce61b12d" />

**<글로벌 스포츠 베팅 시장 규모, 점유율 및 트렌드 분석 보고서 – 산업 개요 및 2033년까지의 전망>**

> 출처: [스포츠 베팅 시장의 규모, 트렌드 및 성장 예측 2033](https://www.databridgemarketresearch.com/ko/reports/global-sports-betting-market)

## 2-3. 프로젝트 필요성

### 1) 신규 이용자 초기 이탈 문제

> 데이터 분석 결과, 신규 가입 이용자 중 상당수가 가입 이후 초기 활동 단계에서 빠르게 이탈하는 경향을 보이고 있습니다.
>
> 온라인 베팅 플랫폼은 신규 이용자 확보를 위해 마케팅 및 운영 비용이 투입되는 구조이기 때문에, 초기 이탈은 단순 이용 감소를 넘어 고객 획득 비용 회수 실패로 이어질 수 있습니다.

### 2) 합법 스포츠 베팅 제도의 운영 안정성 확보 필요

> 합법 스포츠 베팅 제도는 불법 사행 시장을 줄이고, 사행 수요를 제도권 내에서 관리하기 위해 운영되고 있습니다. 그러나 합법 플랫폼에 유입된 신규 이용자가 초기에 이탈할 경우, 해당 수요가 다시 비공식 영역으로 이동할 가능성이 존재합니다.
> 특히, 온라인 환경에서는 플랫폼 간 이동이 용이하기 때문에 초기 이탈은 단순 활동 감소를 넘어서 합법 플랫폼의 이용자 기반 약화로 이어질 수 있습니다.

<img width="1050" height="167" alt="image" src="https://github.com/user-attachments/assets/69a49e9a-925b-4480-8ef4-920e93ce2a62" />

> 출처: [관점이 있는 뉴스 프레미안](https://www.pressian.com/pages/articles/21257)


## 2-5. 프로젝트 목표

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

# 3. 기술 스택

### Language
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Data Analysis
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white)

### Machine Learning
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

---

# 4. WBS 및 폴더 구조

<img width="1130" height="748" alt="image" src="assets/WBS.png" />



# Project Structure

```
04_ml_projects/
├── .github/
│   └── pull_request_template.md
├── .gitignore
├── assets/
│   └── *.png
├── data/
│   ├── processed/
│   │   └── ljh_preprocessed.csv
│   └── raw/
│       ├── sports_gb_F.csv
│       ├── sports_gb_L.csv
│       └── sports_gb_total.csv
├── docs/
│   ├── ai_training_report_template.md
│   ├── git_strategy.md
│   └── model_manager.md
├── models/
│   ├── knn_churn_v1/
│   │   └── report_pyh_knn.md
│   └── xgb/
│       ├── reports_ljh_xgboost.md
│       ├── xgb_meta.json
│       └── xgb.joblib
├── notebooks/
│   ├── chj/
│   │   └── 03_modeling.ipynb
│   ├── ljh/
│   │   ├── 01_preprocessing.ipynb
│   │   ├── 02_EDA.ipynb
│   │   └── 03_modeling.ipynb
│   ├── psy/
│   │   ├── 01_preprocessing.ipynb
│   │   ├── 02_EDA.ipynb
│   │   ├── 02_EDA_refine.ipynb
│   │   ├── 03_Modeling1.ipynb
│   │   ├── psy_RF_Churn_v1.joblib
│   │   ├── psy_RF_Churn_v1_meta.json
│   │   └── sports_gb_total_final.csv
│   └── pyh/
│       └── 03_modeling.ipynb
├── src/
│   └── utils/
│       ├── __init__.py
│       ├── model_manager.py
│       └── plot_config.py
├── main.ipynb
├── README.md
└── requirements.txt
```

## 디렉토리 설명

| 디렉토리          | 설명                                       |
| ----------------- | ------------------------------------------ |
| `assets/`         | README에 사용되는 이미지 파일              |
| `data/raw/`       | 원본 데이터 (수정 금지)                    |
| `data/processed/` | 전처리 완료된 데이터                       |
| `docs/`           | 프로젝트 가이드 및 문서                    |
| `models/`         | 학습 완료된 모델 파일 및 보고서            |
| `notebooks/`      | 팀원별 분석 노트북 (chj / ljh / psy / pyh) |
| `src/utils/`      | 공용 유틸리티 모듈                         |

---

# 5. 데이터 수집 및 출처

## 5-1. 인터넷 베팅 사이트 데이터

- 출처: [The Transparency Project - Data Download](http://www.thetransparencyproject.org/download_index.php)
- 설명: 2005년 2월 1일부터 2006년 8월 31일까지의 온라인 스포츠 베팅 기록

## 5-2. 고객 이탈 선정 기준

- 출처: [Help - General Information - Does bwin charge inactivity fees? (UK)](https://help.bwin.com/en/general-information/account/inactive)
- 설명: 13개월 연속으로 베팅을 하지 않으면 '비활성'으로 분류

---

# 6. 데이터 전처리 및 통합

## 원본 데이터 명세

### (1) `sports_gb_F.csv`

| Feature 명  | 설명                                                                         |
| :---------- | :--------------------------------------------------------------------------- |
| **UserID**  | 등록 시 bwin이 부여한 숫자형 ID                        |
| **DateBet** | 베팅 활동 날짜 (YYYY-MM-DD)                            |
| **StakeF**  | 경기 전 스포츠 베팅에 사용된 총 금액 (유로)        |
| **WinF**    | 경기 전 스포츠 베팅을 통해 획득한 총 당첨금 (유로) |
| **BetsF**   | 해당 일자에 배치된 경기 전 스포츠 베팅 횟수        |

### (2) `sports_gb_L.csv`

| Feature 명  | 설명                                                                         |
| :---------- | :--------------------------------------------------------------------------- |
| **UserID**  | 등록 시 bwin이 부여한 숫자형 ID                        |
| **DateBet** | 베팅 활동 날짜 (YYYY-MM-DD)                            |
| **StakeL**  | 경기 중 스포츠 베팅에 사용된 총 금액 (유로)        |
| **WinL**    | 경기 중 스포츠 베팅을 통해 획득한 총 당첨금 (유로) |
| **BetsL**   | 해당 일자에 배치된 경기 중 스포츠 베팅 횟수        |

### (3) `sports_gb_total.csv`

| Feature 명     | 설명                                            |
| :------------- | :---------------------------------------------- |
| **UserID**     | 등록 시 bwin이 부여한 숫자형 ID                 |
| **CountryID**  | 이용자 거주 국가 코드 (대부분 ISO 3166-1 기준)  |
| **Gender**     | 성별 (0: 여성, 1: 남성)                         |
| **BirthYear**  | 구독자 출생 연도 (1900 ~ 1991)                  |
| **DateReg**    | 구독자 서비스 등록 날짜                         |
| **TimeReg**    | 구독자 서비스 등록 시간 (hh:mm)                 |
| **Date1Dep**   | 계좌에 처음 자금을 예치한 날짜                  |
| **Date1Bet**   | 종목에 상관없이 첫 베팅을 시도한 날짜           |
| **Date1Spo**   | 경기 전 혹은 경기 중 스포츠 베팅을 처음 시도한 날짜 |
| **StakeF/L/A** | 경기 전(F), 경기 중(L), 전체 합계(A) 베팅 금액      |
| **WinF/L/A**   | 경기 전(F), 경기 중(L), 전체 합계(A) 당첨 금액      |
| **BetsF/L/A**  | 경기 전(F), 경기 중(L), 전체 합계(A) 베팅 횟수      |
| **DaysF/L/A**  | 경기 전(F), 경기 중(L), 전체 합계(A) 베팅 발생 일수 |

## 통합 데이터 명세

<img width="1356" height="307" alt="image" src="https://github.com/user-attachments/assets/99c25b6f-36b2-465c-b45c-e7d2291f8b25" />

```
# 0. UserID : 각 유저별 고유 인덱스. 식별자 컬럼이므로 모델링 시 제외 검토
# 1. CountryID : 국가 코드
# 2. Gender : 성별 코드
# 3. BirthYear : 출생연도. 결측치(2개) 처리 후 int 변환, age/age_group 파생에 사용
# 4. DateReg : 가입일
# 5. TimeReg : 가입시간
# 6. Date1Dep : 첫 입금일
# 7. Date1Bet : 첫 베팅일, 스포츠 이외에도 다른 변수 포함
# 8. Date1Spo : 첫 스포츠 베팅일
# 9. StakeF : 경기 전(F) 베팅금액
# 10. StakeL : 경기 중(L) 베팅금액
# 11. StakeA : 전체(A) 베팅금액
# 12. WinF : 경기 전(F) 베팅 당첨금액
# 13. WinL : 경기 중(L) 당첨금액
# 14. WinA : 전체(A) 당첨/수익금액
# 15. BetsF : 경기 전(F) 베팅 건수
# 16. BetsL : 경기 중(L) 베팅 건수
# 17. BetsA : 전체(A) 베팅 건수
# 18. DaysF : 경기 전(F) 활동일수
# 19. DaysL : 경기 중(L) 활동일수
# 20. DaysA : 전체(A) 활동일수
```

## 6-1. age_group 그룹화
기준년도: 2006년
10대:0 / 20대:1 / 30대:2 / 40대:3 / 50대:4 / 60대:5 / 70대:6 / 80대:7 / 90대이상:8

<img width="330" height="318" alt="image" src="https://github.com/user-attachments/assets/d4d3c51d-9eb2-4003-8bed-c866e49c3725" />

## 6-2. hit_days 계산
`hit_days = ` **적중일수 파생 변수**
- **Fixed/Live**: 각 유형별 독립 계산
- **Total**: F + L 일별 합산 후 판정
  
<img width="330" height="453" alt="image" src="https://github.com/user-attachments/assets/18a6b2d3-66ef-4986-80fd-6363672f9410" />

## 6-3. win_rate 계산

`win_rate = (Win > Stake) / (Bets > 0)` **승률 파생 변수**
- **Fixed/Live**: 각 유형별 독립 계산
- **Total**: F + L 일별 합산 후 판정

<img width="330" height="448" alt="image" src="https://github.com/user-attachments/assets/b793829d-a25e-4c4d-b716-1207244ee5f9" />

## 6-4. avg_roi 계산

`avg_roi = mean((Win - Stake) / Stake)` where `Bets > 0 && Stake > 0` **평균 ROI 파생 변수**
- 일별 ROI를 먼저 계산한 뒤 유저별 평균 → 베팅 빈도 관계없이 하루하루의 수익률을 동등하게 반영
- **Total**: `_fl_w`(win_rate에서 생성, F + L 일별 합산) 재사용

<img width="330" height="446" alt="image" src="https://github.com/user-attachments/assets/9fecd02e-a804-494c-8a7e-ac946a1815be" />

## 6-5. churn 계산 (고객 이탈)
기준일: 데이터 마지막 날짜 `2006-08-31`
- `0`: 기준일로부터 13개월(395일) 이내 베팅 활동 있음
- `1`: 없음

<img width="330" height="180" alt="image" src="https://github.com/user-attachments/assets/e89538e4-57f0-481a-9d3b-d4fb116ceea6" />

### 이탈률을 13개월로 설정한 근거

<img width="747" height="157" alt="image" src="https://github.com/user-attachments/assets/3d976bed-a391-42a2-af74-bacbc2ba996a" />

> 출처: [Bwin 공식 사이트 도움말 답변](https://help.bwin.com/en/general-information/account/inactive)

## 통합 데이터

<img width="1480" height="265" alt="image" src="https://github.com/user-attachments/assets/24420f93-5c7a-47f4-9837-c9a67fa5f93c" />

## 통합 데이터 컬럼

```
# 0. user_id : 각 유저별 고유 인덱스. 식별자 컬럼이므로 모델링 시 제외 검토
# 1. country_id : 국가 코드. 범주형 변수로 사용
# 2. gender : 성별 코드. 범주형 변수로 사용
# 3. age_group : 출생연도(BirthYear) 기반 연령대 파생 변수. 범주형 변수로 사용
# 4. reg_date : 가입일 전처리 컬럼(str). DateReg/TimeReg 기반으로 정리한 가입일 정보
# 5. first_deposit : 첫 입금일 전처리 컬럼(str). Date1Dep 기반
# 6. first_bet : 첫 베팅일 전처리 컬럼(str). Date1Bet 또는 Date1Spo 기준으로 정리한 컬럼
# 7. fixed_bet_amount : 경기 전(F) 베팅금액. 수치형
# 8. live_bet_amount : 경기 중(L) 베팅금액. 수치형
# 9. total_bet_amount : 전체(T) 베팅금액. 수치형
# 10. fixed_win_amount : 경기 전(F) 베팅 당첨금액. 수치형
# 11. live_win_amount : 경기 중(L) 베팅 당첨금액. 수치형
# 12. total_win_amount : 전체(T) 베팅 당첨/수익금액. 수치형
# 13. fixed_bet_cnt : 경기 전(F) 베팅 건수. count형
# 14. live_bet_cnt : 경기 중(L) 베팅 건수. count형
# 15. total_bet_cnt : 전체(T) 베팅 건수. count형
# 16. fixed_active_days : 경기 전(F) 활동일수. count형
# 17. live_active_days : 경기 중(L) 활동일수. count형
# 18. total_active_days : 전체(T) 활동일수. count형
# 19. fixed_hit_days : 경기 전(F) 적중일수 파생 변수. count형 (활동 없음/분모 없음 구간 결측 가능)
# 20. live_hit_days : 경기 중(L) 적중일수 파생 변수. count형 (결측치 다수 존재)
# 21. total_hit_days : 전체(A) 적중일수 파생 변수. count형 (결측치 다수 존재)
# 22. fixed_win_rate : 경기 전(F) 승률/적중률 파생 변수. 수치형 (분모 0 구간 결측 처리)
# 23. live_win_rate : 경기 중(L) 승률/적중률 파생 변수. 수치형 (분모 0 구간 결측 처리)
# 24. total_win_rate : 전체(T) 승률/적중률 파생 변수. 수치형 (분모 0 구간 결측 처리)
# 25. fixed_avg_roi : 경기 전(F) 평균 ROI 파생 변수. 수치형 (분모 0 구간 결측 처리)
# 26. live_avg_roi : 경기 중(L) 평균 ROI 파생 변수. 수치형 (분모 0 구간 결측 처리)
# 27. total_avg_roi : 전체(T) 평균 ROI 파생 변수. 수치형 (분모 0 구간 결측 처리)
# 28. churn : 이탈 여부 타겟 변수(0/1). 모델 학습용 타겟 컬럼
```
---

# 7. EDA 시각화 및 분석 인사이트

<img src="assets/이탈률 그래프.png" width="1050" alt="고객 이탈 비율 및 빈도 그래프">

- 이탈 여부는 위 데이터셋의 회사인 스포츠 베팅 회사 Bwin의 휴먼 전환 일자를 기준으로 1년 1개월, 395일을 기준으로 정의하였습니다.
    - 출처 | "https://help.bwin.com/en/general-information/account/inactive"

- 관측기간인 2005 ~ 2006.08.31 기간 내에 활동이 존재하는 회원 중, 마지막 날짜를 기준으로 더이상 활동하지 않는 회원 비중이 약 30% 이상으로 나타났습니다.

<img src="assets/churn과 상관관계.png" width="1050" alt="이탈과 변수의 상관관계">


- total_active_days (-0.35)와 fixed_active_days (-0.34)가 가장 강한 음의 상관관계를 보이고 있습니다.
- 수익률(ROI)은 이탈과 상관관계가 거의 없다는 점을 파악하였습니다.(-0.01)

<img src="assets/이탈과 boxplot.png" width="1050" alt="이탈과 변수의 상관관계">


잔존 유저의 중앙값은 약 35일 근처인 반면, 이탈 유저는 10일 미만에 매우 낮게 형성되어 있음을 확인했습니다.

- **이는 대부분의 이탈자가 활동 초기에 서비스를 떠났음을 의미합니다.**

이탈자의 대부분은 적은 금액의 베팅 규모와 베팅 건수를 가지고 있음을 확인했습니다.

- **이는 일회성 베팅의 의미가 강하다는 것을 의미합니다.**

수익률 및 승률은 잔존 유저와 큰 차이가 없다는 점을 확인했습니다.
- **이는 고객이 돈을 잃어서 떠나기보다, 서비스를 지속하는 것에 대한 흥미나 동기부여가 부족해서 떠난다는 것 임을 예측할 수 있습니다.**

<img src="assets/활동유저비율.png" width="1050" alt='활동유저비율'>

- 활동 일수를 보면, **누적 활동 7일 이내에 약 22%**의 유저가 이탈하며, 누적 활동 30일 시점에는 유저의 절반 이상(57.1%)이 이탈함
**초반 1~7일(누적 활동 기준) 안에 유저에게 확실한 서비스와 동기를 줘야하는 것을 시사함**

<img src="assets/베팅 유형별 이탈률.png" width="1050" alt='베팅 유형별 이탈률'>

- 여러 유형의 베팅을 하는 유저가 가장 낮은 이탈률(19.5%)을 기록함
**이는 다양한 서비스가 고객의 이탈을 막는 것으로 볼 수 있음**

## 분석 결과 요약
EDA 분석 결과, 고객의 이탈은 단순한 금전적 손실(ROI)보다는 **초기 활동의 연속성**과 **서비스 경험의 다양성 부재**에서 나타난다는 점을 파악할 수 있습니다.

---

# 8. 머신러닝 성능 분석 및 인사이트 도출


## 모델 성능 비교

| 모델                |  Accuracy  | Precision  |   Recall   | **F1-Score** |  ROC-AUC   |
| ------------------- | :--------: | :--------: | :--------: | :----------: | :--------: |
| KNN                 |   0.7777   |   0.6208   |   0.5359   |    0.5753    |   0.7993   |
| Logistic Regression |   0.8056   |   0.6500   |   0.6500   |    0.6500    |   0.8638   |
| Random Forest       |   0.8138   | **0.7100** |   0.5725   |    0.6333    |   0.7402   |
| **XGBoost**         | **0.8392** |   0.6780   | **0.8144** |  **0.7400**  | **0.9180** |

> **최우수 모델: XGBoost** — Recall, F1-Score, ROC-AUC 전 항목 1위

---

## 모델별 주요 결과 및 인사이트

### 1. KNN

| 구분      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------- | -------- | --------- | ------ | -------- | ------- |
| Baseline  | 0.7777   | 0.6208    | 0.5359 | 0.5753   | 0.7993  |
| 최적 모델 | 0.7777   | 0.6208    | 0.5359 | 0.5753   | 0.7993  |

- `GridSearchCV(scoring='recall', cv=5)` — n_neighbors 1~13 탐색, 최적값 = **5** (기본값과 동일)
- 하이퍼파라미터 단독 조정으로는 성능 개선 불가 — **전처리(스케일링·결측치·피처)가 성능을 결정**
- 거리 기반 알고리즘 특성상 고차원·결측 데이터에 취약
- **한계**: 이탈 고객 10명 중 약 5명을 놓침 (Recall 0.536)

---

### 2. Logistic Regression

| 구분      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --------- | -------- | --------- | ------ | -------- | ------- |
| Baseline  | 0.8057   | 0.6600    | 0.6500 | 0.6500   | 0.8636  |
| 최적 모델 | 0.8056   | 0.6500    | 0.6500 | 0.6500   | 0.8638  |

- `GridSearchCV(scoring='f1', cv=5)` — C값 탐색, 최적 C=**100** (lbfgs)
- 튜닝 후 성능 변화 거의 없음 → **선형 모델의 표현력 한계치**
- Train F1 0.66 / Val F1 0.65 — 안정적, 과적합 없음
- **핵심 피처**: `total_active_days` 계수 **-9.52**

---

### 3. Random Forest

| 구분                   | Accuracy | Recall | F1-Score | ROC-AUC | Train Score       |
| ---------------------- | -------- | ------ | -------- | ------- | ----------------- |
| Baseline (max_depth=7) | 0.8081   | 0.5325 | 0.5325   | 0.7241  | —                 |
| hyperopt 자동 튜닝     | 0.8140   | 0.5728 | 0.6337   | 0.7405  | **0.92 (과적합)** |
| 수동 튜닝 (최종)       | 0.8138   | 0.5725 | 0.6333   | 0.7402  | **0.87 (개선)**   |

**hyperopt 탐색 공간**: n_estimators, max_depth, min_samples_split, min_samples_leaf
**최적 파라미터**: `n_estimators=350, max_depth=19, min_samples_split=12, min_samples_leaf=1`

**과적합 탐지 및 해결 과정**

> hyperopt가 max_depth=19로 깊은 트리를 선택 → Train Score 0.92 vs Test Score 0.81 (11%p 차이)
> `min_samples_split: 12 → 30` 상향 조정만으로 Train Score 0.87로 낮추면서 Test 성능 유지

---

### 4. XGBoost

| 구분      | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Baseline  | 0.8261     | 0.6487     | 0.8306     | 0.7284     | 0.9154     |
| 최적 모델 | **0.8392** | **0.6780** | **0.8144** | **0.7400** | **0.9180** |
| 개선폭    | +0.0131    | +0.0293    | -0.0162    | +0.0116    | +0.0026    |

**최적 하이퍼파라미터** (`RandomizedSearchCV, 5-fold CV, n_iter=50`):

```python
best_params = {
    'n_estimators':    500,
    'max_depth':       3,        # Baseline(6) 대비 단순화 → 과적합 방지
    'learning_rate':   0.1,
    'min_child_weight': 5,
    'reg_alpha':       0.1,      # L1 규제
    'reg_lambda':      5.0,      # L2 규제
    'scale_pos_weight': 2.56,    # 불균형 처리 (잔존:이탈 = 2.56:1)
}
```

**핵심 기법**:

| 기법           | 설정                       | 효과                                    |
| -------------- | -------------------------- | --------------------------------------- |
| 불균형 처리    | `scale_pos_weight=2.56`    | 이탈 클래스 손실 가중치 상향            |
| Early Stopping | `early_stopping_rounds=30` | 최적 트리 수 283 자동 결정, 과적합 방지 |
| 임계값 최적화  | `threshold: 0.5 → 0.5724`  | Precision +0.029, F1 +0.012             |

> CV ROC-AUC 0.9181 ≈ Test ROC-AUC 0.9180 — **일반화 성능 매우 안정적**

---

## 피처 중요도 — 공통 인사이트

### 1. 활동 지속성

> "얼마나 꾸준히 플랫폼에 접속하느냐"가 이탈을 결정하는 중요 요소

| 피처                | LR 계수   | RF 중요도       | 방향          |
| ------------------- | --------- | --------------- | ------------- |
| `total_active_days` | **-9.52** | **0.204 (1위)** | 증가 → 이탈 ↓ |
| `fixed_active_days` | -0.36     | 0.147 (2위)     | 증가 → 이탈 ↓ |
| `live_active_days`  | -0.36     | —               | 증가 → 이탈 ↓ |

### 2. 베팅 빈도

> "금액"보다 "횟수"가 이탈 방지에 더 중요한 요소

| 피처            | RF 중요도   | 방향          |
| --------------- | ----------- | ------------- |
| `fixed_bet_cnt` | 0.095 (3위) | 증가 → 이탈 ↓ |
| `total_bet_cnt` | 0.081 (4위) | 증가 → 이탈 ↓ |

### 3. 적중 경험 및 ROI

> 경기를 맞추는 경험이 플랫폼 재방문 동기를 부여

| 피처             | RF 중요도   | 방향               |
| ---------------- | ----------- | ------------------ |
| `fixed_hit_days` | 0.048 (7위) | 적중 경험 → 이탈 ↓ |
| `fixed_avg_roi`  | 0.038 (9위) | ROI 양호 → 이탈 ↓  |

---

## 최종 모델 선정 및 비즈니스 시사점

### 1. 운영 적용 모델: XGBoost (threshold=0.5724)

| 지표      | 수치      | 의미                                |
| --------- | --------- | ----------------------------------- |
| Recall    | 0.814 | 이탈 고객 10명 중 약 8명을 사전에 식별 |
| Precision | 0.678     | 이탈 예측 고객 중 약 68%가 실제 이탈 |
| F1-Score  | 0.740 | 탐지 성능과 정확도의 균형 확보 |
| False Positive | 1,006  | 일부 오탐은 마케팅 개입 비용으로 관리 가능 |
| False Negative | 484     | 미탐 고객 최소화를 위한 지속 개선 필요 |

### 2. 이탈 조기 탐지 기준

```
우선순위 경보 조건:
1. total_active_days : 최근 7일 이상 접속 없음 또는 활동 증가 속도 둔화
2. fixed_bet_cnt / live_bet_cnt : 주간 베팅 횟수 30% 이상 감소
3. hit_days 연속 0 : 3주 이상 수익 경험(적중) 부재
```

### 3. 세그먼트별 대응 전략

| 위험 유형  | 주요 신호           | 대응 방향                    |
| -------------- | ------------------- | ----------------------------- |
| 활동일 감소    | 접속 간격 증가      | 재참여 알림 및 이벤트 안내 |
| 이용 빈도 감소 사용자 | 베팅 횟수 급감 | 소액 인센티브 및 프로모션 제공     |
| 성과 경험 부족 사용자 | 장기간 미적중 | 가이드 콘텐츠 및 추천 제공          |
| 신규 초기 위험군 | 가입 초기 활동 저조 | 온보딩 경험 강화              |

---

* **이탈 유저의 전형적 특성: 일회성 베팅 성향**
    * 이탈 유저의 베팅 규모와 건수는 잔존 유저 대비 현저히 낮으며, 누적 활동 중앙값이 10일 미만인 점을 고려할 때, 일회성 베팅 성향이 강하다는 것을 볼 수 있습니다.
    * 수익률(ROI) 및 승률 지표는 잔존 유저와 유의미한 차이를 보이지 않았으며(상관계수 -0.01), 이는 고객이 경제적 손실 때문이 아니라 서비스 지속에 대한 흥미나 동기부여 부족으로 유추할 수 있습니다.

* **초기 30일의 중요성**
    * 누적 활동 7일 이내에 유저의 22.4%, 30일 시점에는 57.1%가 이탈하는 것을 확인했습니다.
    * 이는 활동 초기 1주일 이내에 유저에게 강한 흥미와 접속 동기를 제공하여 **'누적 활동 30일'** 달성이 최우선 과제임을 의미합니다.

* **서비스 다양성**
    * 베팅 유형별 분석 결과, Fixed Only 유형(이탈률 43.0%)보다 Fixed와 Live를 병행하는 **Mixed 유형의 이탈률(19.5%)이 2.2배 이상 낮게** 나타났습니다.
    * 이는 다양한 서비스를 복합적으로 이용하는 유저일수록 이탈 가능성이 현저히 낮아짐을 의미합니다.

 ## 시연 화면
<img width="3420" height="2214" alt="image" src="https://github.com/user-attachments/assets/52e59e1b-6657-4b57-9652-2576de8a8b31" />



---
# 한줄회고

### - 박수영:
> 3개의 데이터 셋을 이용하는 과정에서 fixed, Live 그리고 총 유저 요약 데이터셋까지 파생변수를 만들기 위해 fixed, Live 파일을 이용하였다.  그 과정에서 데이터셋의 구조가 행마다 하루를 단위로 베팅이 기록되다 보니,  하루에 여러번 베팅하면 그것의 결과를 파악할 수 없었다. 따라서 파생변수가 유저의 세세한 행동을 기록하지 못하여 정확한 수치를 측정할 수 없었다는 점에서 아쉬움이 남았다. 또한 하이퍼 파라미터 튜닝 과정에서, 튜닝 이후에 수동으로 파라미터를 튜닝하는 것이 성능이 더 잘나오는 경우가 있어 어떤 경우에 수동으로 튜닝해야 하는지 파악하기 힘들었다.

### - 박영훈:
> knn을 포함한 Logistic Regression, random forest, XGBoost의 다양한 모델을 활용한 머신 러닝을 진행하였는데, knn 모델은 특히 전처리에 매우 민감하다는 점을 파악했습니다. 특히, 하이퍼 파리미터 튜닝으로 한 방식도 미세한 상승 결과를 보였지만, 전처리 진행으로서 산출 된 결과값이 유의미한 결과치를 보며 중요성을 다시 한번 파악할 수 있었습니다.

### - 임정희:
> 성능을 향상시키기 위하여 임계값 조정, 최적의 하이퍼파라미터를 찾는 과정에서 어려움을 겪었습니다. 이를 해결하기 위해 수업 시간에 배운 것들을 적용하는 과정에서 분산되어 있더 개념을 정리할 수 있었습니다.

### - 최하진:
> 모델 튜닝보다 데이터의 품질이 성능의 기저를 형성한다는 본질을 체감하며 전처리의 중요성을 깊이 이해하게 되었습니다. 단순 수치 조정을 넘어 모델의 구조적 한계를 넘어서기 위해, 정교한 피처 엔지니어링과 앙상블 기법을 연계하여 실무적 가치를 지닌 고성능 모델로 디벨롭할 계획입니다.
