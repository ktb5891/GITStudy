# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 데이터 정제 및 준비

# ## 누락된 데이터 처리하기

import numpy as np
import pandas as pd

string_data = pd.Series(['aardvark','artichoke',np.nan,'avocado'])
string_data

string_data.isnull()

string_data[0] = None

string_data.isnull()

# ### 누락된 데이터 골라내기

from numpy import nan as NA

data = pd.Series([1,NA,3.5,NA,7])

data.dropna()

data[data.notnull()]

data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],[NA,NA,NA], [NA,6.5,3.]])

cleaned = data.dropna()

data

cleaned

data.dropna(how='all')

data[4] = NA

data

data.dropna(axis=1, how='all')

df = pd.DataFrame(np.random.randn(7,3))

df.iloc[:4,1] = NA

df.iloc[:2,2] = NA

df

df.dropna()

df.dropna(thresh=2)

# ### 결측치 채우기

df.fillna(0)

df.fillna({1:0.5, 2:0})

_ = df.fillna(0, inplace=True)

df

df = pd.DataFrame(np.random.randn(6,3))

df.iloc[2:,1] = NA
df.iloc[4:,2] = NA

df

df.fillna(method='ffill')

df.fillna(method = 'ffill',limit=2)

data = pd.Series([1.,NA,3.5,NA,7])

data.fillna(data.mean())

# ## 데이터 변형

# ### 중복 제거하기

data = pd.DataFrame({'k1':['one','two']*3 +['two'], 'k2':[1,1,2,3,3,4,4]})

data

# #### 중복인 row를 확인 duplicated()

data.duplicated()

# #### duplicated 에서 False인 값 return

data.drop_duplicates()

data['v1'] = range(7)
data.drop_duplicates(['k1'])

data.drop_duplicates(['k1','k2'],keep='last')

# ### 함수나 매핑을 이용해서 데이터 변형하기

data = pd.DataFrame({'food':['bacon','pulled pork','bacon','Pastrami','corned beef','Bacon','pastrami','honey ham','nova lox'], 
                    'ounces':[4,3,12,6,7.5,8,3,5,6]})

data

meat_to_animal = {
    'bacon':'pig',
    'pulled pork':'pig',
    'pastrami':'cow',
    'corned beef':'cow',
    'honey ham':'pic',
    'nova lox':'salmon'
}

lowercased = data['food'].str.lower()

lowercased

data['animal'] = lowercased.map(meat_to_animal)
data

data['food'].map(lambda x: meat_to_animal[x.lower()])

# ### 값 치환하기

data = pd.Series([1., -999.,2.,-999.,-1000.,3.])
data

data.replace(-999,np.nan)

data.replace([-999,-1000],np.nan)

data.replace([-999,-1000],[np.nan,0])

data.replace({-999:np.nan, -1000:0})

# ### 축(Row) 색인(index) 이름 바꾸기

data = pd.DataFrame(np.arange(12).reshape((3,4)),
                   index=['Ohio','Colorado','New York'],
                   columns=['one','two','three','four'])

transform = lambda x:x[:4].upper()

data.index.map(transform)

data.index = data.index.map(transform)

data

data.rename(index=str.title, columns=str.upper)

data.rename(index={'OHIO':'INDIANA'},columns={'three':'peekaboo'})

data.rename(index={'OHIO':'INDIANA'},inplace=True)
data

# ### 개별화와 양자화

ages = [20,22,25,27,21,23,37,31,61,45,41,32]
bins = [18,25,35,60,100]

cats = pd.cut(ages, bins)
cats

cats.codes

# #### 4개의 그룹으로 분류됨

cats.categories

pd.value_counts(cats)

# #### right = False를 사용하여 구간 값 재설정

pd.cut(ages,[18,26,36,61,100], right=False)

# #### labes로 그룹의 이름 설정

group_names = ['Youth','YoungAdult','MiddleAged','Senoir']
pd.cut(ages,bins,labels=group_names)

data = np.random.rand(20)
pd.cut(data,4,precision=2)

data = np.random.randn(1000) # 정규 분포

cats = pd.qcut(data, 4)
cats

pd.value_counts(cats)

# #### %로 그룹 지정

pd.qcut(data,[0,0.1,0.5,0.9,1.])

# ### 특잇값을 찾고 제외하기

data = pd.DataFrame(np.random.randn(1000,4))
data.describe()

col = data[2]
col[np.abs(col) > 3] # 절대값 3 초과 값

data[(np.abs(data) > 3).any(1)] # 절대값 3 초과 값을 포함하는 row값

data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()

np.sign(data).head()

# ### 치환과 임의 샘플링

df = pd.DataFrame(np.arange(5*4).reshape((5,4)))

sampler = np.random.permutation(5)

sampler

df

df.take(sampler)

df.sample(n=3)

choices = pd.Series([5,7,-1,6,4])

draws = choices.sample(n=10, replace = True)
draws

# ### 표시자 / 더미 변수 계산하기

df = pd.DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})

pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'], prefix='key')

df_with_dummy = df[['data1']].join(dummies)

df_with_dummy

mnames = ['movie_id','title','genres']

# #### https://grouplens.org/datasets/movielens/ 데이터 받아오기

movies = pd.read_table('ch07/movies.csv',sep=',',header=0,names=mnames)
movies[:10]

all_genres = []
for x in movies.genres:
    all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)

genres

zero_matrix = np.zeros((len(movies),len(genres)))
dummies = pd.DataFrame(zero_matrix,columns=genres)

# +
gen = movies.genres[0]

gen.split('|')
# -

dummies.columns.get_indexer(gen.split('|'))

for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i,indices] = 1

movies_windic = movies.join(dummies.add_prefix('Genre_'))

movies_windic.iloc[0]

np.random.seed(12345)
values = np.random.rand(10)
values

bins = [0,0.2,0.4,0.6,0.8,1]

pd.get_dummies(pd.cut(values, bins))

# ## 문자열 다루기

# ### 문자열 객체 메서드

val = 'a,b, guido'

val.split(',')

pieces = [x.strip() for x in val.split(',')]
pieces

first,second,third = pieces
first+'::'+second+'::'+third

'::'.join(pieces)

'guido' in val

val.index(',')

val.find(':')

# +
# val.index(':')
# Error 발생
# -

val.count(',')

val.replace(',','::')

val.replace(',','')

# ### 정규 표현식

import re

# #### \s+로 탭,스페이스,개행문자로 나누어 문자열 분리 

text = "foo bar\t baz \tqux"
re.split('\s+', text) 

regex = re.compile('\s+')

regex.split(text)

regex.findall(text)

# +
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

regex = re.compile(pattern, flags=re.IGNORECASE)
# -

regex.findall(text)

m = regex.search(text)
m

text[m.start():m.end()]

# #### 정규 표현 패턴이 문자열의 시작점에서부터 일치하는지 검사

print(regex.match(text))

# #### 찾은 패턴을 주어진 문자열로 치환해 새 문자열 return

print(regex.sub('REDACTED',text))

# #### ()안에 일치하는 패턴만 뽑아 가져오기

pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags = re.IGNORECASE)

m = regex.match('wesm@bright.net')
m.groups()

regex.findall(text)

print(regex.sub(r'Username: \1, Domain: \2, Shuffix: \3', text))

# ### pandas의 벡터화된 문자열 함수

data = {'Dave':'dave@google.com', 'Steve':'steve@gmail.com','Rob':'rob@gmail.com','Wes':np.nan}
data = pd.Series(data)
data

data.isnull()

data.str.contains('gmail')

pattern

data.str.findall(pattern, flags=re.IGNORECASE)

matches = data.str.match(pattern,flags=re.IGNORECASE)
matches

data.str[:5]


