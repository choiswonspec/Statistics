# Statsmodels



statsmodels(“스탯츠모델즈”라고 읽는다) 패키지는 추정 및 검정, 회귀분석, 시계열분석 등의 기능을 제공하는 파이썬 패키지다. 기존에 R에서 가능했던 다양한 회귀분석과 시계열분석 방법론을 그대로 파이썬에서 이용할 수 있다. 다음은 statsmodels 패키지가 제공하는 기능의 일부다.

- 예제 데이터셋
- 검정 및 모수추정
- 회귀분석
- 선형회귀
- 강건회귀
- 일반화 선형모형
- 혼합효과모형
- 이산종속변수
- 시계열 분석
- SARIMAX 모형
- 상태공간 모형
- 벡터 AR 모형
- 생존분석
- 요인분석

## statsmodels 패키지 소개

- statsmodels 는 통계 분석을 위한 Python 패키지다.

- 전통적인 통계와 계량경제학 알고리즘 포함

- statsmodels 메인 웹사이트 : http://www.statsmodels.org

  user-guide :  https://www.statsmodels.org/stable/user-guide.html

## statsmodels에서 제공하는 통계 분석 기능

- 통계 (Statistics)
  - 각종 검정(test) 및 추정 기능
  - 커널 밀도 추정
  - Generalized Method of Moments
- 회귀 분석 (Linear Regression)
  - 선형 모형 (Linear Model)
  - 일반화 선형 모형 (Generalized Linear Model)
  - 강인 선형 모형 (Robust Linear Model)
  - 선형 혼합 효과 모형 (Linear Mixed Effects Model)
  - ANOVA (Analysis of Variance)
  - Discrete Dependent Variable (Logistic Regression 포함)
- 시계열 분석 (Time Series Analysis) ..
  - AR, ARMA, ARIMA, VAR 및 기타 모델
  - ARMA/ARIMA Process
  - Vector ARMA Process
- 비모수 기법 : 커널밀도추정, 커널회귀
- 통계 모델 결과의 시각화
- 인자를 위한 불확실성 예측치와 p 값을 제공
- scikit-learn은 좀 더 예측에 초점을 맞추고 있다.

특히 선형 회귀분석의 경우 R-style 모형 기술을 가능하게 하는 patsy 패키지를 포함하고 있어 기존에 R을 사용하던 사람들도 쉽게 statsmodels를 쓸 수 있게 되었다.

- https://patsy.readthedocs.org/en/latest/

## statsmodels를 사용하여 선형 회귀 분석을 수행 예시

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 데이터 로드
dat = sm.datasets.get_rdataset("Guerry", "HistData").data
dat.tail()

>>>
dept	Region	Department	Crime_pers	Crime_prop	Literacy	Donations	Infants	Suicides	MainCity	...	Crime_parents	Infanticide	Donation_clergy	Lottery	Desertion	Instruction	Prostitutes	Distance	Area	Pop1831
81	86	W	Vienne	15010	4710	25	8922	35224	21851	2:Med	...	20	1	44	40	38	65	18	170.523	6990	282.73
82	87	C	Haute-Vienne	16256	6402	13	13817	19940	33497	2:Med	...	68	6	78	55	11	84	7	198.874	5520	285.13
83	88	E	Vosges	18835	9044	62	4040	14978	33029	2:Med	...	58	34	5	14	85	11	43	174.477	5874	397.99
84	89	C	Yonne	18006	6516	47	4276	16616	12789	2:Med	...	32	22	35	51	66	27	272	81.797	7427	352.49
85	200	NaN	Corse	2199	4589	49	37015	24743	37016	2:Med	...	81	2	84	83	9	25	1	539.213	8680	195.41
5 rows × 23 columns
# 회귀 분석 
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

# 결과 출력
print(results.summary())

>>>
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                Lottery   **R-squared:                       0.348**
Model:                            OLS   Adj. R-squared:                  0.333
Method:                 Least Squares   F-statistic:                     22.20
Date:                Thu, 21 Apr 2016   Prob (F-statistic):           1.90e-08
Time:                        03:20:40   Log-Likelihood:                -379.82
No. Observations:                  86   AIC:                             765.6
Df Residuals:                      83   BIC:                             773.0
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [95.0% Conf. Int.]
-----------------------------------------------------------------------------------
Intercept         246.4341     35.233      6.995      **0.000**       176.358   316.510
Literacy           -0.4889      0.128     -3.832      **0.000**        -0.743    -0.235
np.log(Pop1831)   -31.3114      5.977     -5.239      **0.000**       -43.199   -19.424
==============================================================================
Omnibus:                        3.713   Durbin-Watson:                   2.019
Prob(Omnibus):                  0.156   Jarque-Bera (JB):                3.394
Skew:                          -0.487   Prob(JB):                        0.183
Kurtosis:                       3.003   Cond. No.                         702.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

## ststsmodels 내장 datasets

statsmodels 패키지의 개발 목표 중 하나는 기존에 R을 사용하여 통계 분석 및 시계열 분석을 하던 사용자가 파이썬에서 동일한 분석을 할 수 있도록 하는 것이다.

따라서 R에서 제공하던 명령어 뿐만 아니라 데이터셋도 동일하게 제공하기 위해 Rdatasets이라는 프로젝트를 통해 R에서 사용하던 1000개 이상의 표준 데이터셋을 사용할 수 있도록 지원한다. 자세한 사항은 다음 프로젝트 홈페이지에서 확인할 수 있다.

- https://github.com/vincentarelbundock/Rdatasets

다음은 위 프로젝트에서 제공하는 데이터셋의 목록이다.

- http://vincentarelbundock.github.io/Rdatasets/datasets.html

이 목록에 있는 데이터를 가져오려면 우선 "Package"이름과 "Item"을 알아낸 후 다음에 설명하는 `get_rdataset` 명령을 이용한다.

```
get_rdataset(item, [package="datasets"])
```

`item`과 `package` 인수로 해당 데이터의 "Package"이름과 "Item"을 넣는다. "Package"이름이 `datasets`인 경우에는 생략할 수 있다. 이 함수는 인터넷에서 데이터를 다운로드 받으므로 인터넷에 연결되어 있어야 한다. 이렇게 받은 데이터는 다음과 같은 속성을 가지고 있다.

- `package`: 데이터를 제공하는 R 패키지 이름
- `title`: 데이터 이름 문자열
- `data`: 데이터를 담고 있는 데이터프레임
- `__doc__`: 데이터에 대한 설명 문자열. 이 설명은 R 패키지의 내용을 그대로 가져온 것이므로 예제 코드가 R로 되어 있어 파이썬에서는 사용할 수 없다.

`get_rdataset` 명령으로 받을 수 있는 몇가지 예제 데이터를 소개한다.

## boston housing 예제 Data 불러오기

```python
import os
import pandas as pd 
import numpy as np

# 현재경로 확인
os.getcwd()
>>>
'C:\\\\Users\\\\csw31'
# 데이터 불러오기
from sklearn.datasets import load_boston
boston = load_boston()
boston_x = pd.DataFrame(boston.data, columns=boston.feature_names)
target = pd.DataFrame(boston.target, columns=["MEDV"])

boston_ = pd.concat([boston_x, target], axis=1)
boston_.tail()  # boston raw data

>>>
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
501	0.06263	0.0	11.93	0.0	0.573	6.593	69.1	2.4786	1.0	273.0	21.0	391.99	9.67	22.4
502	0.04527	0.0	11.93	0.0	0.573	6.120	76.7	2.2875	1.0	273.0	21.0	396.90	9.08	20.6
503	0.06076	0.0	11.93	0.0	0.573	6.976	91.0	2.1675	1.0	273.0	21.0	396.90	5.64	23.9
504	0.10959	0.0	11.93	0.0	0.573	6.794	89.3	2.3889	1.0	273.0	21.0	393.45	6.48	22.0
505	0.04741	0.0	11.93	0.0	0.573	6.030	80.8	2.5050	1.0	273.0	21.0	396.90	7.88	11.9
```

타겟 데이터 1978 보스턴 주택 가격 506개 타운의 주택 가격 중앙값 (단위 1,000 달러)

- CRIM: 범죄율, INDUS: 비소매상업지역 면적 비율, NOX: 일산화질소 농도, RM: 주택당 방 수, LSTAT: 인구 중 하위 계층 비율, B: 인구 중 흑인 비율, PTRATIO: 학생/교사 비율, ZN: 25,000 평방피트를 초과 거주지역 비율, CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0, AGE: 1940년 이전에 건축된 주택의 비율, RAD: 방사형 고속도로까지의 거리, DIS: 직업센터의 거리, TAX: 재산세율

## 상관계수 확인

```python
# Target 빼고 나머지 컬럼 따로 저장 
boston_data = boston_.drop(['MEDV'],axis=1)
# 상관계수를 통해 다중공선성 확인
boston_data.corr()
# 히트맵을 그려서 상관계수 시각적으로 표현
import seaborn as sns;
import matplotlib.pyplot as plt
#cmap = sns.light_palette("darkgray", as_cmap=True)
plt.figure(figsize=(20,10))
sns.heatmap(boston_data.corr(), annot=True )
plt.show()
```

## 산점도를 통해 다중공산성 확인

```python
cols = ["MEDV", "RM", "AGE", "RAD"]
sns.pairplot(data = boston_, vars=cols)
plt.show()

'''sns.pairplot(data = df, vars = numerical_columns,  hue='Sex', palette="rainbow", height=5,)
plt.show()'''
```

## VIF (Variance Inflation Factors, 분산팽창요인)를 통한 다중공산성 확인

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(boston_data.values, i) for i in range(boston_data.shape[1])]
vif["features"] = boston_data.columns
vif
```

변수를 삭제해가면서 비교하면 다중공산성이 조금씩 줄어든다.

## Statsmodels 를 이용한 다중선형회귀분석

### 모델이 아닌 분석용... R의 형태를 가져옴

만들어진 모델이 최종적으로 정말 쓸만한가 아니면 더 좋은 모델을 만드는 방향으로 가는게 나은지를 진단하는 목적

fitted 모델은 다음의 속성을 가진다

- params: 가중치 벡터
- resid: 잔차 벡터

<formula 에 사용하는 기호>

- 1, 0	바이어스(bias, intercept) 추가 및 제거
- 설명 변수 추가
- 설명 변수 제거
- :	상호작용(interaction)
- a*b = a + b + a:b
- /	a/b = a + a:b

I() 연산자를 활용하면 다항회귀(polynomial regression)도 할 수 있다.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols(formula = '방문빈도 ~ 거래기간 + 총매출액 + 쿠폰사용횟수', data = data)
result = model.fit()
result.summary()
result.params
>>>
const      36.459488
CRIM       -0.108011
ZN          0.046420
INDUS       0.020559
CHAS        2.686734
NOX       -17.766611
RM          3.809865
AGE         0.000692
DIS        -1.475567
RAD         0.306049
TAX        -0.012335
PTRATIO    -0.952747
B           0.009312
LSTAT      -0.524758
dtype: float64
```

## 로지스틱 회귀분석

- 종속변수가 binary

- 모델의 회귀계수와 오즈비를 구해 독립변수가 분류 결정에 미치는 영향의 정도를 알아보기 위해 다른 방식의 로지스틱회귀분석을 진행해보자.

  logit이란 변수에 합격여부를 종속변수로 하는 데이터를 넣고,

  앞서 넣은 독립변수 x값도 입력한 뒤 적합 시켜준다.

```python
import statsmodels.api as sm
logit = sm.Logit(data[['합격여부']],x) #로지스틱 회귀분석 시행  y, x 형태
result = logit.fit()
result.summary2()
```

- 결과 해석 방법 :

  1. 먼저 유의 확률을 보고 유의한 변수들만 남기고 나머지 변수 제외하여 돌린다.
  2. 편회귀계수값(Coef.)의 부호를 통해 종속변수에 미치는 영향의 방향 파악 가능 .

  - 양수라면 1로 분류할 확률을 늘려주고 음수라면 0으로 분류할 확률을 높여준다.

  1. 오즈비가 1을 기준으로 큰지 작은지를 파악하여 종속변수에 미치는 영향의 방향 파악 가능

독립변수가 두개 이상 있을 때는 다른 독립변수를 일정한 값으로 고정한 경우의 오즈비로 해석된다. 아무런 관계없을 때 오즈비는 1이다. 1에서 멀리 떨어질수록 종속변수와의 관계가 강하다는 뜻이다. 즉, 종속변수 여부에 큰 영향을 준다는 뜻이다.  오즈비는 1을 기준으로 영향을 판단하므로, 오즈비가 10인 경우와 0.1인 경우는 종속변수에 영향을 주는 강도가 같다. 입시점수 변수의 경우, 극도로 1에 가까운 값으로 나타난다. 따라서, 입시점수는 합격여부에 별다른 영향을 주지않았음(관계 없음)을 알 수 있다. 반면, 학점의 경우는 각 오즈비가 0.61로 1과 떨어져있으므로 합격여부에 영향을 미쳤음을 알 수 있다. 독립변수가 수치형일 경우 또 다르게 오즈비해석을 할 수 있다. 학점이 1단위 증가하면 합격할 확률이 0.61배 증가한다는 뜻으로도 해석할 수 있다.

- 입시점수 1.001563
- 학점 0.617389
- dtype: float64

```python
# 오즈비 구하기
np.exp(result.params)
```

# ANOVA

Analysis of Variance models containing anova_lm for ANOVA analysis with a linear OLSModel, and AnovaRM for repeated measures ANOVA, within ANOVA for balanced data.

분산분석(ANOVA)는 전체 그룹간의 평균값 차이가 통계적 의미가 있는지 판단하는데 유용한 도구 입니다. 하지만 정확히 어느 그룹의 평균값이 의미가 있는지는 알려주지 않습니다. 따라서 추가적인 사후분석(Post Hoc Analysis) 이 필요합니다.

## 일원분산분석(One-way ANOVA)

종속변인은 1개이며, 독립변인의 집단도 1개인 경우입니다. 한가지 변수의 변화가 결과 변수에 미치는 영향을 보기 위해 사용

파이썬에서 One-way ANOVA 분석은 scipy.stats이나 statsmodel 라이브러리를 이용해서 할 수 있습니다.

statsmodel 라이브러리가 좀 더 많고 규격화된 정보를 제공합니다.

```python
import pandas as pd
import urllib
import matplotlib.pyplot as plt

# url로 데이터 얻어오기
url = '<https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/altman_910.txt>'
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# Sort them into groups, according to column 1
group1 = data[data[:,1]==1,0]
group2 = data[data[:,1]==2,0]
group3 = data[data[:,1]==3,0]

# matplotlib plotting
plot_data = [group1, group2, group3]
ax = plt.boxplot(plot_data)
plt.show()
from statsmodels.formula.api import ols

# 경고 메세지 무시하기
import warnings
warnings.filterwarnings('ignore')

df = pd.DataFrame(data, columns=['value', 'treatment'])    

# the "C" indicates categorical data
model = ols('value ~ C(treatment)', df).fit()

print(anova_lm(model))
```

일원분산분석 예시

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f6b4e99-3cd7-414f-9881-a8970a13248b/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/af553ddf-30ad-4dae-abf4-34ee924aa3a1/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e247365e-4161-4d52-8998-01668f2fa1c4/Untitled.png)

## 이원분산분석(two-way ANOVA)

독립변인의 수가 두 개 이상일 때 집단 간 차이가 유의한지를 검증하는 데 사용합니다. 상호작용효과(Interaction effect), 한 변수의 변화가 결과에 미치는 영향이 다른 변수의 수준에 따라 달라지는지를 확인하기 위해 사용

```python
inFile = 'altman_12_6.txt'
url_base = '<https://raw.githubusercontent.com/thomas-haslwanter/statsintro_python/master/ipynb/Data/data_altman/>'
url = url_base + inFile
data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')

# Bring them in dataframe-format
df = pd.DataFrame(data, columns=['head_size', 'fetus', 'observer'])
# df.tail()

# 태아별 머리 둘레 plot 만들기
df.boxplot(column = 'head_size', by='fetus' , grid = False)
```

분산분석으로 상관관계

```python
from statsmodels.stats.anova import anova_lm

# 교호작용도 표현 가능
formula = 'head_size ~ C(fetus) + C(observer) + C(fetus):C(observer)' 
lm = ols(formula, df).fit()
print(anova_lm(lm))
```

P-value 가 0.05 이상 입니다, 따라서 귀무가설을 기각할 수 없고. 측정자와 태아의 머리둘레값에는 연관성이 없다고 할 수 있습니다. 측정하는 사람이 달라도 머리 둘레값은 일정하는 것이죠.

결론적으로 초음파로 측정하는 태아의 머리둘레값은 믿을 수 있다는 것입니다.

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

moore = sm.datasets.get_rdataset("Moore", "carData", cache=True) # load data
data = moore.data

data = data.rename(columns={"partner.status":  "partner_status"}) # make name pythonic

moore = ols('conformity ~ C(fcategory, Sum)*C(partner_status, Sum)',data=data)
fitted = moore.fit()

# Type 2 ANOVA DataFrame 테이블 형태의 결과
table = sm.stats.anova_lm(moore, typ=2) 
print(table)
```

이원분산분석 예시

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8211fc7c-cc50-4968-8765-a60a2bc3e321/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c4650a21-fe29-4ebd-87e2-d9731dd6c091/Untitled.png)

## 다원배치분산분석(Multivariate ANOVA)(MANOVA)

- 연속형 종속변수가 두개 이상일 경우

https://www.youtube.com/watch?v=FYM3NtNjZCI

```python
from statsmodels.multivariate.manova import MANOVA
print(MANOVA.from_formula('수치형종속변수1+수치형종속변수2+수치형종속변수3 ~ 범주형독립변수', data = df).mv_test())
```