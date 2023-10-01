    
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 
from sklearn.preprocessing import LabelEncoder


datacomb = pd.read_csv("https://raw.githubusercontent.com/jeffdarmawan/it5006-team-16/main/data_allthreeyears_combined.csv")
# print(datacomb.columns)
included_columns = ['Age', 'Education level_attainedOrGGtoAttain', 'Coding Experience (in years)', 'Years in ML', 
                    'Job_Salary', 'Money Spent on ML/Cloud Computing', 'Times used TPU']
cols = [col for col in datacomb.columns if col in included_columns]

all_dict = {}
first = True
all = pd.DataFrame()
for col in cols:
    col_df = pd.DataFrame()
    col_df[col] = datacomb[col].unique()
    # col_df.columns = [col]
    # all[col] = datacomb[col].unique()
    col_df[col+'_encoded'] = LabelEncoder().fit_transform(col_df[col]) # random values provided by sklearn
    all = pd.concat([all, col_df], axis=1)
    print(all.shape)


    # ans_dict = {}
    # label_encoder = LabelEncoder()
    # ans = datacomb[col].unique()
    # encoded_data = label_encoder.fit_transform(ans)

    # # sorted_ans = sorted(datacomb[col].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
    # for i, ans in enumerate(ans):
    #     ans_dict[ans] = encoded_data[i]
    # all_dict[col] = ans_dict
all.to_csv('all_ans.csv')
print(all)

print(datacomb['Money Spent on ML/Cloud Computing'].unique())
# What happens after this is that we label the ordinal data based its intrinsic order
# For example, the ordinal data for education level is as follows:
# 1. No formal education past high school
# 2. Some college/university study without earning a bachelor’s degree
# 3. Bachelor’s degree
# 4. Master’s degree
# 5. Doctoral degree
# 6. Professional degree
# 7. I prefer not to answer

