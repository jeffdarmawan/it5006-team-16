# Milestone 1.2
# Milestone 1.2 with single-multi and single-single selection
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

datacomb = pd.read_csv("data_allthreeyears_combined.csv")
datacomb = datacomb.rename(columns={'Gender - Selected Choice': 'Gender', 'Job_title - Selected Choice': 'Job_Title'})
# Southeast Asia countries
# source: https://en.wikipedia.org/wiki/Southeast_Asia
sea_countries = ["Brunei","Cambodia","East Timor","Indonesia","Laos","Malaysia","Myanmar","Philippines","Singapore","Thailand","Vietnam"]
datacomb = datacomb[datacomb['Location'].isin(sea_countries)]

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

# 3. Import ordinal data encoding
ord_encodings = pd.read_csv('ordinal_encodings.csv')
included_columns = ['Age', 'Education level_attainedOrGGtoAttain', 'Coding Experience (in years)', 'Years in ML', 
                    'Job_Salary', 'Money Spent on ML/Cloud Computing', 'Times used TPU']
nominal_features_order_dict = {}
for col in included_columns:
    nominal_features_order_dict[col] = dict(zip(ord_encodings[col], ord_encodings[col+'_encoded']))

# st.title('Milestone 1.2')
st.header('Filtered Data and Plot based on SEA countries')

# Sidebar filters
st.sidebar.header('Select Columns to Display')

# x_axis are multi-select columns
x_axis = st.sidebar.selectbox('X-axis',['Job_Salary','Job_JobScope','Coding Experience (in years)',
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
 'Job_Title',
 'Location',
 'Money Spent on ML/Cloud Computing',
 'Times used TPU',
 'Years in ML'])

# y_axis are single-select columns
y_axis = st.sidebar.selectbox('Y-axis',['Coding Experience (in years)','Age','Job_Salary', 
 'Education level_attainedOrGGtoAttain',
 'Gender',
 'Job_EmployerUsingML?',
 'Job_Title',
 'Location',
 'Money Spent on ML/Cloud Computing',
 'Times used TPU',
 'Years in ML'])

is_row_select_col_2_ordinal = False
is_col_select_col_1_ordinal = False
col_order_dict = {}
row_order_dict = {}

if x_axis in nominal_features_order_dict.keys():
    is_col_select_col_1_ordinal = True
    col_order_dict = nominal_features_order_dict[x_axis]

if y_axis in nominal_features_order_dict.keys():
    is_row_select_col_2_ordinal = True
    row_order_dict = nominal_features_order_dict[y_axis]

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
x_axis_is_multi = False
y_axis_is_multi = False

if x_axis in multi_select_cols:
    x_axis_is_multi = True

if y_axis in multi_select_cols:
    y_axis_is_multi = True

# check if the selected options are ordinal columns
lst_of_ordinal_col_names = ['Age', 'Education level_attainedOrGGtoAttain', 'Coding Experience (in years)', 'Years in ML','Job_EmployerUsingML?','Job_Salary', 'Money Spent on ML/Cloud Computing', 'Times used TPU']

x_axis_is_ordinal = False
y_axis_is_ordinal = False

if x_axis in lst_of_ordinal_col_names:
    x_axis_is_ordinal = True

if y_axis in lst_of_ordinal_col_names:
    y_axis_is_ordinal = True


# Create checkboxes in the sidebar
st.sidebar.title('Select the year(s) of interest:')
year2020 = st.sidebar.checkbox('Year 2020', value=True)
year2021 = st.sidebar.checkbox('Year 2021', value=True)
year2022 = st.sidebar.checkbox('Year 2022', value=True)

selected_years = []
if year2020:
    selected_years.append(2020)
if year2021:
    selected_years.append(2021)
if year2022:
    selected_years.append(2022)

# Altering the dataframe based on the selection made by the us

# 3. '''If there are 2 selected_columns ''' ----------------------------------
if len(x_axis) > 0 and len(y_axis) > 0:

# 3.1 '''If both are single-select columns ''' ----------------------------------
    if ((x_axis_is_multi == False) and (y_axis_is_multi == False)):

        df_salary_exp = datacomb[['year',x_axis, y_axis,'Location']]

        df_salary_exp = df_salary_exp.dropna(subset=['year',x_axis, y_axis], how = 'any')

        # Altering the dataframe to only consider the selected_years
        df_salary_exp = df_salary_exp[df_salary_exp['year'].isin(selected_years)]

        # the code below shows a dataframe with 'x_axis' and 'y_axis'
        df_salary_exp_count_data = df_salary_exp.groupby([y_axis, x_axis]).size().reset_index(name='count')

        # turn the dataframe above into a pivot table for plotting on heatmap
        df_salary_exp_heatmap_data = df_salary_exp_count_data.pivot_table(index = y_axis, columns = x_axis, values='count', fill_value=0)

        df_salary_exp_heatmap_data_1 = df_salary_exp_heatmap_data.loc[sort_pivot_table_row(df_salary_exp_heatmap_data), sort_pivot_table_col(df_salary_exp_heatmap_data)]


        fig, ax = plt.subplots(figsize=(20, 12))
        # sns.heatmap(df_salary_exp_heatmap_data_1, annot=True, fmt='d', cmap=cmap, cbar=True, xticklabels=salary_order, yticklabels=job_experience_order)
        sns.heatmap(df_salary_exp_heatmap_data_1, annot=True, fmt='d', cmap=cmap, cbar=True, xticklabels = sort_pivot_table_col(df_salary_exp_heatmap_data), yticklabels=sort_pivot_table_row(df_salary_exp_heatmap_data))
                                                                                                                            
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('Heatmap: '+ x_axis +' vs. ' + y_axis +'(Count)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        st.pyplot(fig)

        total_counts = df_salary_exp_heatmap_data_1.sum(axis=1)
        percentage_table = df_salary_exp_heatmap_data_1.div(total_counts, axis=0) * 100
        for column in percentage_table.columns:
            fig, ax = plt.subplots(figsize=(20, 12))
            plt.bar(percentage_table.index, percentage_table[column])
            plt.xlabel(y_axis)
            plt.ylabel('Percentage')
            plt.title(f'{y_axis} Distribution for {column}')
            plt.xticks(rotation=45)  # Rotate x-axis labels 
            plt.tight_layout()
            
            for p in ax.patches:
                ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
            st.pyplot(fig)

        # print("percentage_table.columns")
        # print(percentage_table.columns)
        # print("percentage_table.index")
        # print(percentage_table.index)
        # print("percentage_table.index")
        # print(percentage_table)
        # ce = np.array(percentage_table.columns)
        # g = sns.FacetGrid(percentage_table, col=ce[:,np.newaxis], col_wrap=5)
        # g.map_dataframe(sns.barplot, x=percentage_table.index, y=y_axis)
        # st.pyplot(g.fig)
        


    # 3.2 '''If there is one single-select and one multi-select column ''' ----------------------------------
    if ((x_axis_is_multi == True) and (y_axis_is_multi == False)) or ((x_axis_is_multi == False) and (y_axis_is_multi == True)):
        if (x_axis_is_multi == True):
            multi_selected_col = x_axis
            single_selected_col = y_axis
            
        if (y_axis_is_multi == True):
            multi_selected_col = y_axis
            single_selected_col = x_axis

        # Select relevant columns, including popular programming languages
        multi_selected_columns = [col for col in datacomb.columns if multi_selected_col in col]

        df_salary_exp = datacomb[['year', single_selected_col] + multi_selected_columns]

        # Drop rows with missing values in any of the multi_selected_columns (programming language) columns
        df_salary_exp = df_salary_exp.dropna(subset= multi_selected_columns, how='all')
        df_salary_exp = df_salary_exp.dropna(subset= [single_selected_col], how='any')

        cols_to_be_unpivoted = [col for col in df_salary_exp.columns if multi_selected_col in col]

        unpivoted_df_salary_exp = pd.melt(df_salary_exp, id_vars = single_selected_col, value_vars = cols_to_be_unpivoted, \
                                        var_name = 'Unpivoted Multi-Select Col', value_name='True for Unpivoted Multi-Select Col?')

        unpivoted_df_salary_exp = unpivoted_df_salary_exp[unpivoted_df_salary_exp['True for Unpivoted Multi-Select Col?'].notna()]

        unpivoted_df_salary_exp = unpivoted_df_salary_exp.drop(columns=['True for Unpivoted Multi-Select Col?'])

        df_salary_count_data = unpivoted_df_salary_exp.groupby([single_selected_col, 'Unpivoted Multi-Select Col']).size().reset_index(name='count')

        # generate pivot table for heatmap
        df_salary_exp_heatmap_data = df_salary_count_data.pivot_table(index=single_selected_col, columns='Unpivoted Multi-Select Col', values='count', fill_value=0)

        # sort the pivot table if there are ordinal features in the pivot table, before plotting it on the heatmap
        df_salary_exp_heatmap_data_1 = df_salary_exp_heatmap_data.loc[sort_pivot_table_row(df_salary_exp_heatmap_data), sort_pivot_table_col(df_salary_exp_heatmap_data)]\

        fig, ax = plt.subplots(figsize=(20, 12))
        sns.heatmap(df_salary_exp_heatmap_data_1, annot=True, fmt='d', cmap=cmap, cbar=True, xticklabels = sort_pivot_table_col(df_salary_exp_heatmap_data), yticklabels=sort_pivot_table_row(df_salary_exp_heatmap_data))
                                                                                                                            
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('Heatmap: ' + x_axis + ' vs. '+y_axis+'(Count)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        total_counts = df_salary_exp_heatmap_data_1.sum(axis=1)
        percentage_table = df_salary_exp_heatmap_data_1.div(total_counts, axis=0) * 100
        for column in percentage_table.columns:
            fig, ax = plt.subplots(figsize=(20, 12))
            plt.bar(percentage_table.index, percentage_table[column])
            plt.xlabel(y_axis)
            plt.ylabel('Percentage')
            plt.title(f'{y_axis} Distribution for {column}')
            plt.xticks(rotation=45)  # Rotate x-axis labels 
            plt.tight_layout()
            
            for p in ax.patches:
                ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
            
            st.pyplot(fig)