import pandas as pd
# read_csv로 불러오기
df = pd.read_csv('ch06/ex1.csv')
print(df)

print('\n')
# read_table로 불러오기
print(pd.read_table('ch06/ex1.csv', sep=','))