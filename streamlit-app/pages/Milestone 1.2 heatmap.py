# Milestone 1.2

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# colouring for heatmap----------------------------------
import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list(
    'custom_cmap',
[(0, 'white'), (0.5, 'blue'), (1, 'red')])
# =======================================================


st.set_option('deprecation.showPyplotGlobalUse', False)

datacomb = pd.read_csv("https://raw.githubusercontent.com/jeffdarmawan/it5006-team-16/main/data_allthreeyears_combined.csv")
datacomb = datacomb.rename(columns={'Gender - Selected Choice': 'Gender', 'Job_title - Selected Choice': 'Job_Title'})
# sea_countries = ["Brunei","Cambodia","East Timor","Indonesia","Laos","Malaysia","Myanmar","Philippines","Singapore","Thailand","Vietnam"]
datacomb = datacomb[(datacomb['Location']=='Indonesia') | (datacomb['Location']=='Malaysia') | (datacomb['Location']=='Philippines') | (datacomb['Location']=='Singapore') | (datacomb['Location']=='Thailand')]
# print(datacomb.head())  # Check the first few rows of the loaded data

# cleaning some columns
datacomb['Job_Salary'].replace('300,000-500,000', '300,000-499,999', inplace = True)
datacomb['Coding Experience (in years)'].replace('1-2 years','1-3 years', inplace = True)
datacomb = datacomb.rename(columns={'Gender - Selected Choice': 'Gender', 'Job_title - Selected Choice': 'Job_Title'})



# 1.'''Identifying Multi-Select Columns''' ----------------------------------

multi_select_cols = set() # this store the names of the multi-select cols

for col in datacomb.columns:
    if " - " in col:
        multi_select_qn = col.split(" - ")[0]
        multi_select_cols.add(multi_select_qn)
        
# 2. '''Identifying Single-Select Columns''' ----------------------------------

single_select_cols = set() # this store the names of the single-select cols

for col in datacomb.columns:
    if " - " not in col and col != 'year' and col != 'Time spent on survey':
        single_select_cols.add(col)

'''Streamlit app----------------------------------------------------------------'''

st.title('Milestone 1.2')
st.header('Filtered Data and Plot based on SEA countries (Indonesia, Malaysia, Philippines, Singapore, Thailand)')

# Sidebar filters
st.sidebar.header('Select Columns to Display')

# selected_column_1 are multi-select columns
selected_column_1 = st.sidebar.selectbox('Select Columns',['Job_Salary','Job_JobScope','Coding Experience (in years)',
 'Learning platforms tried',
 'Popular BI tool brands',
 'Popular Cloud Computing Platform Brand',
 'Popular Cloud Computing Product Brand',
 'Popular Computer Vision Methods',
 'Popular IDEs',
 'Popular ML Algorithms',
 'Popular ML frameworks',
 'Popular ML product brand',
 'Popular NLP Methods',
 'Popular auto ML product brand',
 'Popular data product brands used (Databases, Warehouses, Lakes)',
 'Popular hosted notebook products',
 'Popular media sources for Data Science',
 'Popular programming language',
 'Popular tools to monitor ML/Experiments',
 'Age',
 'Education level_attainedOrGGtoAttain',
 'Gender',
 'Job_EmployerUsingML?',
 'Job_No.OfDSTeamMember',
 'Job_Title',
 'Location',
 'Money Spent on ML/Cloud Computing',
 'Times used TPU',
 'Years in ML'])

# selected_column_2 are single-select columns
selected_column_2 = st.sidebar.selectbox('Select Columns',['Coding Experience (in years)','Age','Job_Salary', 
 'Education level_attainedOrGGtoAttain',
 'Gender',
 'Job_EmployerUsingML?',
 'Job_No.OfDSTeamMember',
 'Job_Title',
 'Location',
 'Money Spent on ML/Cloud Computing',
 'Times used TPU',
 'Years in ML'])

# ''' For sorting of heatmap's axis for ordinal features'''------------------------------------

# column order dictionary for ordinal features-----------------
nominal_features_order_dict = {'Job_Salary': {'$0-999':1,
'1,000-1,999':2,\
'2,000-2,999':3,\
'3,000-3,999':4,\
'4,000-4,999':5,\
'5,000-7,499':6,\
'7,500-9,999':7,\
'10,000-14,999':8,\
'15,000-19,999':9,\
'20,000-24,999':10,\
'25,000-29,999':11,\
'30,000-39,999':12,\
'40,000-49,999':13,\
'50,000-59,999':14,\
'60,000-69,999':15,\
'70,000-79,999':16,\
'80,000-89,999':17,\
'90,000-99,999':18,\
'100,000-124,999':19,\
'125,000-149,999':20,\
'150,000-199,999':21,\
'200,000-249,999':22,\
'250,000-299,999':23,\
'300,000-499,999':24,\
'$500,000-999,999':25,\
'> $500,000':26,\
'>$1,000,000':27},\
'Coding Experience (in years)': {'I have never written code':1,'< 1 years':2, '1-3 years':3, '3-5 years':4, '5-10 years':5, '10-20 years':6,'20+ years':7}}
# =========================================================

is_row_select_col_2_ordinal = False
is_col_select_col_1_ordinal = False
col_order_dict = {}
row_order_dict = {}

if selected_column_1 in nominal_features_order_dict.keys():
    is_col_select_col_1_ordinal = True
    col_order_dict = nominal_features_order_dict[selected_column_1]
    st.write(col_order_dict)

if selected_column_2 in nominal_features_order_dict.keys():
    is_row_select_col_2_ordinal = True
    row_order_dict = nominal_features_order_dict[selected_column_2]
    st.write(row_order_dict)

def sort_pivot_table_row(pivot_table):
    if is_row_select_col_2_ordinal:
        row_order = []
        row_order_new = []
        for i in pivot_table.index.unique():
            row_order.append((row_order_dict[i],i))
        row_order.sort()
        for v,k in row_order:
            row_order_new.append(k)
        return row_order_new
    else:
        return list(pivot_table.index.unique()) # anyhow give back order cos it doesn't matter

def sort_pivot_table_col(pivot_table):
    if is_col_select_col_1_ordinal:
        col_order = []
        col_order_new = []
        for i in pivot_table.columns.unique():

            col_order.append((col_order_dict[i],i))
        col_order.sort()
        for v,k in col_order:
            col_order_new.append(k)
        return col_order_new
    else:
        return list(pivot_table.columns.unique())


# ==================================================================================================

# check if the selected options are multi-select columns
selected_column_1_is_multi = False
selected_column_2_is_multi = False

if selected_column_1 in multi_select_cols:
    selected_column_1_is_multi = True

if selected_column_2 in multi_select_cols:
    selected_column_2_is_multi = True

# check if the selected options are ordinal columns
lst_of_ordinal_col_names = ['Age', 'Education level_attainedOrGGtoAttain', 'Coding Experience (in years)', 'Years in ML','Job_EmployerUsingML?','Job_Salary', 'Money Spent on ML/Cloud Computing', 'Times used TPU']

selected_column_1_is_ordinal = False
selected_column_2_is_ordinal = False

if selected_column_1 in lst_of_ordinal_col_names:
    selected_column_1_is_ordinal = True

if selected_column_2 in lst_of_ordinal_col_names:
    selected_column_2_is_ordinal = True


# Create checkboxes in the sidebar
st.sidebar.title('Select the year(s) of interest:')
year2020 = st.sidebar.checkbox('Year 2020', value=False)
year2021 = st.sidebar.checkbox('Year 2021', value=True)
year2022 = st.sidebar.checkbox('Year 2022', value=True)

selected_years = []
if year2020:
    selected_years.append(2020)
if year2021:
    selected_years.append(2021)
if year2022:
    selected_years.append(2022)

print(selected_column_1)
print(selected_column_2)

# Altering the dataframe based on the selection made by the user
# (ref) df_salary_exp = datacomb[['year','Job_Salary', 'Coding Experience (in years)','Location']]
# (ref) df_salary_exp = df_salary_exp.dropna(subset=['year','Job_Salary', 'Coding Experience (in years)'], how = 'any')
st.write(datacomb.head(2))

# 3. '''If there are 2 selected_columns ''' ----------------------------------
if len(selected_column_1) > 0 and len(selected_column_2) > 0:

# 3. '''If both are single-select columns ''' ----------------------------------
    if ((selected_column_1_is_multi == False) and (selected_column_2_is_multi == False)):

        df_salary_exp = datacomb[['year',selected_column_1, selected_column_2,'Location']]

        df_salary_exp = df_salary_exp.dropna(subset=['year',selected_column_1, selected_column_2], how = 'any')

        # Altering the dataframe to only consider the selected_years
        df_salary_exp = df_salary_exp[df_salary_exp['year'].isin(selected_years)]
        st.write(df_salary_exp)
        # the code below shows a dataframe with 'selected_column_1' and 'selected_column_2'
        # (ref) df_salary_exp_count_data = df_salary_exp.groupby(['Coding Experience (in years)', 'Job_Salary']).size().reset_index(name='count')
        df_salary_exp_count_data = df_salary_exp.groupby([selected_column_2, selected_column_1]).size().reset_index(name='count')

        # turn the dataframe above into a pivot table for plotting on heatmap
        # (ref) df_salary_exp_heatmap_data = df_salary_exp_count_data.pivot_table(index='Coding Experience (in years)', columns='Job_Salary', values='count', fill_value=0)
        df_salary_exp_heatmap_data = df_salary_exp_count_data.pivot_table(index = selected_column_2, columns = selected_column_1, values='count', fill_value=0)
        # print(df_salary_exp_heatmap_data)

        df_salary_exp_heatmap_data_1 = df_salary_exp_heatmap_data.loc[sort_pivot_table_row(df_salary_exp_heatmap_data), sort_pivot_table_col(df_salary_exp_heatmap_data)]


        fig, ax = plt.subplots(figsize=(20, 12))
        # sns.heatmap(df_salary_exp_heatmap_data_1, annot=True, fmt='d', cmap=cmap, cbar=True, xticklabels=salary_order, yticklabels=job_experience_order)
        sns.heatmap(df_salary_exp_heatmap_data_1, annot=True, fmt='d', cmap=cmap, cbar=True, xticklabels = sort_pivot_table_col(df_salary_exp_heatmap_data), yticklabels=sort_pivot_table_row(df_salary_exp_heatmap_data))
                                                                                                                            
        plt.xlabel('Job Salary')
        plt.ylabel('Coding Experience (in years)')
        plt.title('Heatmap: Job Salary vs. Coding Experience (Count)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        st.pyplot()


# # Filter the DataFrame based on selected years




# # 3. '''If there is one single-select and one multi-select column ''' ----------------------------------
# if ((selected_column_1_is_multi == True) and (selected_column_2_is_multi == False)) or ((selected_column_1_is_multi == False) and (selected_column_2_is_multi == True)):
   
#     if (selected_column_1_is_multi == True):
#         multi_selected_col = selected_column_1
#         single_selected_col = selected_column_2
        
#     if (selected_column_2_is_multi == True):
#         multi_selected_col = selected_column_2
#         single_selected_col = selected_column_1
