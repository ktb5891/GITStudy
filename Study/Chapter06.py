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

# # 데이터 로딩과 저장, 파일형식

import pandas as pd

# ### read_csv로 불러오기

df = pd.read_csv('ch06/ex1.csv')
df

# ### read_table로 불러오기

df = pd.read_table('ch06/ex1.csv', sep=',')
df

# ### 컬럼이름 생성하기(자동)

df = pd.read_csv('ch06/ex1.csv', header = None)
df

# ### 컬럼이름 생성하기(직접)

col = ['a','b','c','d','message']
pd.read_csv('ch06/ex1.csv', names=col)

# ### 메세지 컬럼을 인덱스로 지정하기

pd.read_csv('ch06/ex1.csv', names=col, index_col = 'message')

# ### 계층적 인덱스 지정하기

parsed = pd.read_csv('ch06/csv_mindex.csv',index_col=['key1','key2'])
parsed

# ### 고정된 구분자 없이 공백이나 다른 패턴으로 필드를 구분한 파일

list(open('ch06/ex3.txt'))

# #### https://www.nextree.co.kr/p4327/ 정규표현식 참고

result = pd.read_table('ch06/ex3.txt', sep='\s+') 
result

# ### 문장과 같이 예외처리 해야할 인자 예외처리

list(open('ch06/ex4.csv'))

pd.read_csv('ch06/ex4.csv',skiprows=[0,2,3])

# ### 누락되어 특수하게 처리된 문자 'Null'로 처리하기

list(open('ch06/ex5.csv'))

result = pd.read_csv('ch06/ex5.csv')
result

pd.isnull(result)

result = pd.read_csv('ch06/ex5.csv', na_values=['NULL'])
result

# ### 컬럼별로 NA값으로 지정된 값 NA로 바꾸기

sentinels = {'message':['foo'], 'something':['two']}

pd.read_csv('ch06/ex5.csv',na_values = sentinels)

# ## 텍스트 파일 조금씩 읽어오기

# ### 최대 10줄 출력하기

pd.options.display.max_rows = 10

result = pd.read_csv('ch06/ex6.csv')
result

# ### 첫 5줄만 읽기

pd.read_csv('ch06/ex6.csv',nrows = 5)

# ### 파일을 여러조각으로 나누어 읽기

chunker = pd.read_csv('ch06/ex6.csv', chunksize = 1000)
chunker

# +
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
    
tot = tot.sort_values(ascending=False)
# -

tot[:10]

# ## 데이터를 텍스트 형식으로 기록하기

# ### csv파일로 만들기 ( , 로 구분)

data = pd.read_csv('ch06/ex5.csv')
data

data.to_csv('ch06/out.csv')

list(open('ch06/out.csv'))

# ### ( | 로 구분)

import sys
data.to_csv(sys.stdout, sep='|')

# ### 누락된 값 'NULL'을 삽입

data.to_csv(sys.stdout, na_rep='NULL')

# ### row와 column은 제외하기

data.to_csv(sys.stdout, index=False, header = False)

# ### 일부 또는 직접 지정하기

data.to_csv(sys.stdout, index = False, columns = ['a','b','c'])

# ### Series에서 csv파일로 저장하기

import numpy as np

dates = pd.date_range('16/5/2021',periods = 7)

ts = pd.Series(np.arange(7), index = dates)

ts.to_csv('ch06/tseries.csv')

list(open('ch06/tseries.csv'))

# ## 구분자 형식 다루기

list(open('ch06/ex7.csv'))

# ### 파이썬 내장 csv 모듈 이용하기

# +
import csv
f = open('ch06/ex7.csv')

reader = csv.reader(f)
# -

for line in reader:
    print(line)

# ### 해당 파일을 단위 리스트로 저장

with open('ch06/ex7.csv') as f:
    lines = list(csv.reader(f))

header, values = lines[0], lines[1:]
print(header)
print(values)

data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict


# ### csv 모듈에 있는 csv.Dialect 클래스를 상속받아 다양한 구분자 문자열을 처리하는 새로운 클래스를 정의

# +
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
    
# reader = csv.reader(f, dialect=my_dialect)


# +
# reader = csv.reader(f, delimiter = '|')
# -

# ### csv파일 기록하기

with open('ch06/mydata.csv','w') as f:
    writer = csv.writer(f, dialect = my_dialect)
    writer.writerow(('one','two','three'))
    writer.writerow(('1','2','3'))
    writer.writerow(('4','5','6'))
    writer.writerow(('7','8','9'))

list(open('ch06/mydata.csv'))

# ## JSON 데이터

# ### json 문자열을 파이썬 형태로 변환하기

import json

obj = '''
{"name":"Wes",
"places_lived" : ["United States", "Spain","Germany"],
"pet" : null,
"siblings" : [{"name":"Scott", "age" : 30, "pets" : ["Zeus","Zuko"]},
                {"name":"Katie", "age" : 38, "pets" :["Sixes","Stache","Cisco"]}]
}
'''

result = json.loads(obj)
result

# ### 파이썬 객체를 json형태로 변환하기

asjson = json.dumps(result)

asjson

# ### JSON 객체를 DataFrame으로 변환하기

siblings = pd.DataFrame(result['siblings'], columns = ['name','age','pets'])
siblings

data = pd.read_json('ch06/example.json')
data

print(data.to_json())

print(data.to_json(orient='records'))

# ## XML과 HTML: 웹 스크래핑

# ### 미연방예금보험공사 부도은행 데이터 활용

# #### https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/

colnames = ['Bank Name','City','ST','CERT','Acquiring Institution','Closing Date','Fund']
tables = pd.read_html('ch06/FDIC _ Failed Bank List.html')
len(tables)

failures = tables[0]

failures.columns = colnames

failures.head()

close_timestamps = pd.to_datetime(failures['Closing Date'])
close_timestamps.dt.year.value_counts()

# ## lxml.objecify를 이용해서 xml 파싱하기

# ### getroot 함수를 이용해서 xml파일의 루트 노드에 대한 참조하기

# +
from lxml import objectify

path = 'ch06/Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()
# -

data = []
skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ','DESIRED_CHANGE','DECIMAL_PLACES']
for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)

perf = pd.DataFrame(data)
perf.head()

from io import StringIO
tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()

root

root.get('href')

root.text

# ## 이진 데이터 형식

# ### pickle 직렬화로 데이터를 이진형식으로 저장

frame = pd.read_csv('ch06/ex1.csv')
frame

frame.to_pickle('ch06/frame_pickle')

pd.read_pickle('ch06/frame_pickle')

# ### HDF5 형식 사용하기

frame = pd.DataFrame({'a':np.random.randn(100)})

store = pd.HDFStore('mydata.h5')

store['obj1'] = frame

store['obj1_col'] = frame['a']

store

store['obj1']

store.put('obj2',frame,format='table')

store.select('obj2',where=['index >= 10 and index <= 15'])

store.close()

frame.to_hdf('mydata.h5','obj3',format='table')

pd.read_hdf('mydata.h5','obj3',where=['index < 5'])

# ### MS Excel파일에서 데이터 읽어오기

# #### ex1.csv파일 xlsx로 저장하기

xlsx = pd.ExcelFile('ch06/ex1.xlsx')

pd.read_excel(xlsx, 'ex1')

frame = pd.read_excel('ch06/ex1.xlsx','ex1')
frame

writer = pd.ExcelWriter('ch06/ex2.xlsx')

frame.to_excel(writer, 'ex1')

writer.save()

frame.to_excel('ch06/ex2.xlsx')

# ## 웹 API와 함께 사용하기

import requests

url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)
resp

# +
data = resp.json()

data[0]['title']
# -

# ### 관심있는 column만 가져오기

issues = pd.DataFrame(data, columns=['number','title','labels','state'])
issues

# ## 데이터베이스와 함께 사용하기

# ### 파이썬 내장 sqlite3 드라이버로 SQLite DB 이용하기

import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL, d INTEGER);
"""

con = sqlite3.connect('mydata.sqlite')
con.execute(query)

con.commit()

data = [('Atlanta', 'Georgia', 1.25, 6),
       ('Tallahassee','Florida', 2.6, 3),
       ('Sacramento','Califormia', 1.7, 5)]

stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt,data)

con.commit()

cursor = con.execute('select * from test')

rows = cursor.fetchall()
rows

cursor.description

pd.DataFrame(rows, columns=[x[0] for x in cursor.description])

# ### SQLAlchemy를 사용하여 같은 SQLite DB엦 접속해 데이터 불러오기

import sqlalchemy as sqla

db = sqla.create_engine('sqlite:///mydata.sqlite')

pd.read_sql('select * from test', db)


