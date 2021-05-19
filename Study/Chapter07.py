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

# ### 축 색인 이름 바꾸기

data = pd.DataFrame(np.arange(12).reshape((3,4)),
                   index=['Ohio','Colorado','New York'],
                   columns=['one','two','three','four'])

transform = lambda x:x[:4].upper()

data.index.map(transform)

data.index = data.index.map(transform)

data

data.rename(index=str.title, columns=str.upper)


