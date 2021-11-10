# Scipy



scipy(“사이파이”라고 읽는다) 패키지는 고급 수학 함수, 수치적 미적분, 미분 방정식 계산, 최적화, 신호 처리 등에 사용하는 다양한 과학 기술 계산 기능을 제공한다.

# 통계 검정

집단간의 차이 검정

- 단일 표본 t-검정 (One-sample t-test)
- 독립 표본 t-검정 (Independent-two-sample t-test)
- 쌍체 표본 t-검정 (Paired-two-sample t-test)
- 분산 검정 (Chi squared variance test)
- 단일 표본 z-검정 (One-sample z-test)

# 기초 검정

- 이항 검정 (Binomial test)
- 카이 제곱 검정 (Chi-square test)
- 등분산 검정 (Equal-variance test)
- 정규성 검정 (Normality test)

[검정](https://www.notion.so/66a6de90aad442418424d140b3ad5738)



---





# 집단 간의 차이 검정

## 단일표본 z검정

- 모분산 𝜎2의 값을 정확히 알고 있는 정규분포의 표본에 대해 기댓값을 조사하는 검정방법이다.
- 단일표본 z검정의 경우에는 많이 사용되지 않고 사이파이에 별도의 함수가 준비되어 있지 않으므로 norm 명령의 cdf 메서드를 사용하여 직접 구현해야 한다.

```python
N = 10
mu_0 = 0
np.random.seed(0)
x = sp.stats.norm(mu_0).rvs(N)
x
>>>
array([ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
       -0.97727788,  0.95008842, -0.15135721, -0.10321885,  0.4105985 ])
```

- 단일표본 z검정 함수를 다음처럼 구현할 수 있다. 이 함수는 검정통계량과 유의확률을 튜플형태로 반환한다

```python
def ztest_1samp(x, sigma2=1, mu=0):
    z = (x.mean() - mu) / np.sqrt(sigma2/len(x))
    return z, 2 * sp.stats.norm().sf(np.abs(z))

ztest_1samp(x)
>>>
(2.3338341854824276, 0.019604406021683538)
```

만약 유의 수준이 5%면 유의확률이 1.96%이므로 귀무가설을 기각. 따라서 𝜇≠0이다.

## 일표본 t-검정

- 일표본 t-검정은 단일 집단일 경우, 평균에 대한 가설을 검정하기 위해 사용

  ex) 한국 여성의 평균 키는 161cm로 알려져 있는데 실제 한국 여성의 평균 키가 161cm가 맞는지 검정하고 싶을 때

- 검정통계량 , p-value = stats.ttest_1samp( 해당 집단, 검정 대상 평균 값 )

- 예시 문제

```python
from scipy import stats

H_prop = df[df['color'] == 'H'] # 'color'가 'H'인 것 추출
# 평균 3932에 대한 가설 검정. 일표본 t-검정 실시
static, p_value = stats.ttest_1samp(H_prop['price'], 3932)

# hypo 값에 귀무가설 참 거짓 여부 저장
if pv<0.05:
    hypo = False
else:
    hypo = True

# 검정통계량
print(static)
# p-value
print(p_value)
# 귀무가설 참, 거짓 여부
print(hypo)
>>>
11.988997411117696
7.569973305218302e-33
False
```

## 독립표본 t-검정

- 독립된 두 집단의 평균 차이를 검정하는 기법.

- 반드시 서로 무관한 독립된 두 집단을 사용해야한다.

- 독립표본 t-검정의 경우, 등분산 여부에 따라 결과값이 달라지기 때문에 독립표본 t-검정을 시행 하기 전에 등분산 검정을 시행한 후 그 결과에 따라 독립표본 t-검정을 시행한다.  → equal_var = True or False

- 등분산 검정이란 두 모집단에서 추출한 표본의 분산이 같은 것을 말한다.

  남녀의 평균 키에 차이가 있는지 비교할 때,

  10대와 20대의 평균 수면시간 차이를 비교하고자 할 경우,

  한국인과 미국인의 평균 쌀 섭취량 비교와 같은 경우에 독립표본 t-검정을 사용

- 집단간 등분산(levene, fligner, bartlett) 검정

```python
# 두집단 데이터 추출.. 'color' 가 'F', 'G'인 그룹
F = df[df['color'] == 'F']
G = df[df['color'] == 'G']

# 등분산 검정 levene, fligner, bartlett 시행
# '귀무가설은 두 집단의 분산은 같다' 이다
leve = stats.levene(F['price'], G['price'])
fli = stats.fligner(F['price'], G['price'])
bartlet= stats.bartlett(F['price'], G['price'])

print(leve)
print(fli)
print(bartlet)
>>>
LeveneResult(statistic=53.627886257416655, pvalue=2.511093007074788e-13)
FlignerResult(statistic=37.04347553879807, pvalue=1.155244880009172e-09)
BartlettResult(statistic=47.52732212008511, pvalue=5.424264079418252e-12)
```

귀무가설 기각(p-value <0.05)시, 유의수준하에 ‘F와 G 집단간 분산은 같지 않다’

- 독립표본 t-검정

  귀무가설은 ' 두 집단의 평균은 같다' 이다

  등분산 검정 후, 등분산이 아닌 경우 equl_var = False 를 주면 된다.

```python
t_test_FG =stats.ttest_ind(G['price'], F['price'], equal_var = False)
t_test_FG
>>>
Ttest_indResult(statistic=5.045279980436125, pvalue=4.5670321227041464e-07)
```

독립표본 t검정 시행시 귀무가설 기각(p-value <0.05), 유의수준하에 ‘F와 G 집단간 평균은 같지 않다’

## 쌍체표본 t-검정

- 쌍체표본 t-검정이란 동일한 항목, 사람 또는 물건에 대한 측정 값이 두개인 경우에 사용하는 분석방법이다.

- 이때 분석 대상의 표본은 반드시 대응 되어야한다.

- 만약, 대응되지 않는 표본이라면 결측값이 있다는 뜻이므로 결측값을 처리한 후 분석을 진행해야한다.

- 또한, 대응표본은 시간상 전후의 개념이 있기 때문에 독립된 두 집단일 필요가 없다.

- 쌍체표본 t-검정은 다음과 같은 경우에 사용한다.

  7월이 생일인 고객에게 20%할인 쿠폰을 부여한 후 브랜드만족도 조사를 했을 때 쿠폰 부여 이전과의 차이를 검정할 경우,

  다이어트 중인 사람에게 채식을 시키고 3개월 뒤에 몸무게 변화를 검정하고자 할 경우 등이 있다.

```python
before = data['문화센터도입전만족도']
after = data['문화센터도입후만족도']

#쌍체표본 t-검정 실행
result = stats.ttest_rel(after, before)
# statistic, p_value = stats.ttest_rel(after, before) 이렇게 하면 각각 값을 저장 가능

print('t statistic : %.3f \\np-value : %.3f' % (result))

# p-value가 유의하면 귀무가설 기각하여 전후가 차이있다 결론내릴 수 있다
```

## ANOVA 분산 분석

- 2개의 모집단에 대한 평균을 비교, 분석하는 통계적 기법으로 t-Test를 활용하였다면, 비교하고자 하는 집단이 2개 이상일 경우에는 **분산분석을** 이용
- 설명변수는 범주형 자료이어야 하며, 종속변수는 연속형 자료일 때 2개 이상 집단 간 평균 비교분석에 분산분석(ANOVA) 을 사용
- 분산분석(ANOVA)는 전체 그룹간의 평균값 차이가 통계적 의미가 있는지 판단하는데 유용한 도구다
- 하지만 정확히 어느 그룹의 평균값이 의미가 있는지는 알려주지 않는다
- 따라서 추가적인 사후분석(Post Hoc Analysis) 이 필요하다

### **분산분석의 가정**

**1. 각 모집단은 정규분포여야 하며, 집단 간 분산은 동일해야 한다.**

모집단을 서로 비교하기 위해서는 각 모집단이 좌우대칭인 정규분포여야 한다.

여기서 중요한 것은 3개 이상의 모집단의평균을 비교하기 위한 분석이기 때문에, 모집단끼리 분산이 동일하지 않으면 평균 차이를 구별하기 쉽지 않다는 것이다. 때문에 분산분석에서는 위와 같은 가정을 필요로 하게 된다.

**2. 각 표본들은 독립적으로 추출되어야 한다.**

표본을 구성하는 과정에서 어느 집단이 다른 집단에 영향을 주지 않는 독립성을 지녀야 한다.

**3. 각 표본의 크기는 적절해야 한다**

분석을 진행하기 위해서는 표본의 크기가 충분해야 표본의 개수와 상관없이 분석을 진행할 수 있다.

### 일원분산분석(One-way ANOVA)

- 일원분산분석(one-way ANOVA) 는 2개 이상의 그룹 간 평균의 차이가 존재하는지만을 검정하고 한 가지 요인을 기준으로 집단간의 차이를 조사할 때 시행
- 파이썬에서 One-way ANOVA 분석은 scipy.stats이나 statsmodel 라이브러리를 이용해서 할 수 있다.
- statsmodel 라이브러리가 좀 더 많고 규격화된 정보를 제공한다.
- **샘플 데이터에 결측값이 포함되어 있는 경우, 결측값을 먼저 제거해주고 일원분산분석 검정을 실시해야 한다. 결측값(NAN)이 포함되어 있으면 'NAN'을 반환**

우선, 등분산 검정 실시

```python
D = df[df['color'] == 'D']

levene = stats.levene(F['price'], D['price'], G['price'])
fligner =stats.fligner(F['price'], D['price'], G['price'])
bartlett =stats.bartlett(F['price'], D['price'], G['price'])

print(bartlett)
print(fligner)
print(levene)
>>>
BartlettResult(statistic=289.14364432535103, pvalue=1.634012581050329e-63)
FlignerResult(statistic=494.64591695585733, pvalue=3.881538382653518e-108)
LeveneResult(statistic=118.97521469312785, pvalue=3.557425577381817e-52)
```

정규성 검정 실시

```python
Kstest
anova = stats.f_oneway(F['price'], D['price'], G['price'])
anova
>>>
F_onewayResult(statistic=101.1811790316069, pvalue=1.6513790091285713e-44)
```

- 귀무가설을 기각 (p-value <0.05) 시,  유의수준 하에서 세집단 중 어느 두집단의 평균은 같지 않다(정확한 검정을 위해서는 사후검정실시해야함)

### 이원분산분석(two-way ANOVA)

- 두 가지 요인을 기준으로 집단간의 차이를 조사할 때 시행
- 상호작용효과(Interaction effect), 즉 교호작용, 한 변수의 변화가 결과에 미치는 영향이 다른 변수의 수준에 따라 달라지는지를 확인하기 위해 사용.

### 다원분산분석(multi-way ANOVA)

- 세 가지 이상의 요인을 기준으로 집단간의 차이를 조사할 때

### 다변량 분산분석((multi-variate ANOVA)

- 한 가지 이상의 요인을 기준으로 두 가지 이상의 종속변수에 대해 조사할 때

## 표정리

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a5d4df4a-6dc1-455d-862f-aeacb8d69f3f/_2021-06-15__9.25.53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a5d4df4a-6dc1-455d-862f-aeacb8d69f3f/_2021-06-15__9.25.53.png)

# 기초 검정

https://datascienceschool.net/02 mathematics/09.05 사이파이를 사용한 검정.html

## 두 변수 간 상관 계수와 상관 계수의 p-value 구하기

```python
# pandas corr() 을 이용하여 상관 계수 구하기
corr_by_pandas = df[['carat','price']].corr().iloc[0,1] # (0,0)과 (1,1) 은 자기 자신과의 상관계수
print(corr_by_pandas)
>>>
0.9215913011935697
------------------------------------------------------------------------------------
# scipy stats 로 상관계수와 p_value 까지 구하기
from scipy import stats
corr_by_scipy , p_value = stats.pearsonr(df['carat'],df['price']) #피어슨상관계수
print(corr_by_scipy)
>>>
0.9215913011934769
------------------------------------------------------------------------------------
# p-value
print(p_value)
>>>
0.0
```

## 등분산 검정 (Equal-variance test)

```python
df_f = df.loc[df['color']=='F',:] 
df_g = df.loc[df['color']=='G',:] 
df_d = df.loc[df['color']=='D',:]

import numpy as np
from scipy import stats
result1 = stats.levene(df_f['price'], df_g['price'])
result2 = stats.fligner(df_f['price'], df_g['price'])
result3 = stats.bartlett(df_f['price'], df_g['price'])
print(result1)
print(result2)
print(result3)
```

귀무가설 : 모든 집단의 분산이 같다. 기각시 적어도 하나의 집단의 분산이 다르다.

## 이항 검정 (Binomial test)

- 이항검정은 이항분포를 이용하여 베르누이 확률변수의 모수 𝜇에 대한 가설을 조사하는 검정 방법이다. 사이파이 stats 서브패키지의 binom_test 명령은 이항검정의 유의확률을 계산한다. 디폴트 귀무가설은 𝜇=0.5이다.

```python
scipy.stats.binom_test(x, n=None, p=0.5, alternative='two-sided')
```

- `x`: 검정통계량. 1이 나온 횟수
- `n`: 총 시도 횟수
- `p`: 귀무가설의 𝜇μ값
- `alternative`: 양측검정인 경우에는 `'two-sided'`, 단측검정인 경우에는 `'less'` 또는 `'greater'`

예제

```python
N = 10
mu_0 = 0.5
np.random.seed(0)
x = sp.stats.bernoulli(mu_0).rvs(N)
n = np.count_nonzero(x)
n
>>>
7
```

모수가 0.5인 베르누이 분포라면 가장 가능성이 높은 5가 나와야 하는데 여기에는 7이 나왔다. 그렇다면 이 확률변수의 모수는 0.5가 아니라 0.7일까? 모수가 0.5라는 귀무가설의 신빙성을 확인하기 위해 binom_test 이항검정 명령으로 유의확률을 구하면 약 34%이다.

```python
sp.stats.binom_test(n, N)
>>>
0.3437499999999999
```

유의확률이 높으므로 모수가 0.5라는 귀무가설을 기각할 수 없다.

## 카이 제곱 검정 (Chi-square test)

- 

## 정규성 검정 (Normality test)

- 