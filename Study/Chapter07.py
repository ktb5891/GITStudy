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


