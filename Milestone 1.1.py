# Milestone 1
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# datacomb = pd.read_csv("data_allthreeyears_combined.csv")
datacomb = pd.read_csv("data_allthreeyears_combined_new1_exported.csv")

datacomb = datacomb.rename(columns={'Gender - Selected Choice': 'Gender', 'Job_title - Selected Choice': 'Job_Title'})
# Southeast Asia countries
# source: https://en.wikipedia.org/wiki/Southeast_Asia
# sea_countries = ["Brunei","Cambodia","East Timor","Indonesia","Laos","Malaysia","Myanmar","Philippines","Singapore","Thailand","Vietnam"]
# datacomb = datacomb[datacomb['Location'].isin(sea_countries)]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Filter box to select location
selected_locations = st.multiselect('Select Locations (Optional)', datacomb['Location'].unique())
if selected_locations:
    datacomb = datacomb[datacomb['Location'].isin(selected_locations)]

# Filter box to select Job_Title
selected_job_title = st.multiselect('Select Job Title (Optional)', datacomb['Job_Title'].unique())
if selected_job_title:
    datacomb = datacomb[datacomb['Job_Title'].isin(selected_job_title)]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
        
    
# '''Streamlit app'''

# Sidebar filters
st.sidebar.header('Select Feature to Display')

selected_column = st.sidebar.selectbox('Select Feature',['Job_JobScope',
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
 'Coding Experience (in years)',
 'Education level_attainedOrGGtoAttain',
 'Gender',
 'Job_EmployerUsingML?',
 'Job_Salary',
 'Job_Title',
 'Location',
 'Money Spent on ML/Cloud Computing',
 'Times used TPU',
 'Years in ML'])

                                       
st.header('% Count of the selected feature')


# 3. '''If there is a selected_column ''' ----------------------------------

if len(selected_column) > 0:
    i = selected_column
    # check if the selected_column is a single-select or multi-select column
    
    # 3.1. '''Plotting of Multi-Select Columns'''
    if i in multi_select_cols: 
        mul_cols_to_select = ['year'] + [col for col in datacomb.columns if i in col]

        mul_cols_to_select_df = datacomb[mul_cols_to_select]
        feature_counts = {}


        for column in mul_cols_to_select_df.columns:
            if i in column:
                replacement = i + " - "
                feature = column.replace(replacement, '')
                counts_by_year = mul_cols_to_select_df.groupby(['year', column]).size().unstack().fillna(0)
                feature_counts[feature] = counts_by_year
    
        # Create a DataFrame from the feature_counts dictionary
        feature_counts_df = pd.concat(feature_counts, axis=1)

        # Transpose the DataFrame
        feature_counts_df = feature_counts_df.T


        # Loop through each year and divide each cell value by the count of non-NaN values for that year
        for year in feature_counts_df.columns:
            non_nan_count = len(mul_cols_to_select_df.dropna(subset=mul_cols_to_select_df.columns.difference(['year']), how='all')[mul_cols_to_select_df['year'] == year])
            try:
                feature_counts_df[(year)] = feature_counts_df[(year)].apply(lambda x: x / non_nan_count)
            except:
                feature_counts_df[(year)] = 0
        
        # Create a Matplotlib figure with the desired figure size
        fig, ax = plt.subplots(figsize=(20, 12))

        # Plot the data as a grouped bar chart
        feature_counts_df.plot(kind='bar', stacked=False, ax=ax)
        plt.xlabel(f'{i}',fontsize=16)
        plt.ylabel('% of respondents', fontsize=16)
        plt.title(f'Counts of {i} by {i} and Year')

        # Rename the x-axis labels to show only the feature name
        new_labels = [label[1] for label in feature_counts_df.index]
        ax.set_xticklabels(new_labels, rotation=45, horizontalalignment='right')

        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Display the Matplotlib figure in Streamlit
        st.pyplot(fig)
        # (NEW) ====================
        # Calculating the average across all three years:
        feature_counts_df['Average'] = feature_counts_df.mean(axis=1)
        # Extract the desired columns for the table
        table_data = feature_counts_df[['Average']].reset_index(level =0, drop = True)

        fig, ax = plt.subplots(figsize=(20, 12))
        
        # plot this average as bar chart
        feature_counts_df['Average'].plot(kind='bar', ax=ax)
        plt.xlabel(f'{i}', fontsize=16)
        plt.ylabel('Average % of respondents', fontsize=16)
        plt.title(f'Average Counts of {i} by {i}')
        # Rename the x-axis labels to show only the feature name
        new_labels = [label[1] for label in feature_counts_df.index]
        ax.set_xticklabels(new_labels, rotation=45, horizontalalignment='right')

        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        # Display the Matplotlib figure in Streamlit
        st.pyplot(fig)
        st.write('Average Values:')
        st.table(table_data)# NEW =======

    
    
    # 3.2. '''Plotting of Multi-Select Columns'''
    if i in single_select_cols:
        single_cols_to_select_df = datacomb[['year',i]]
        
        single_col_count_by_yr = single_cols_to_select_df.groupby(['year',str(i)]).size().unstack()    # Create a dataframe where the columns = age range, rows = years
    
        # Transpose the DataFrame (columns = years, row = age range)
        single_col_count_by_yr = single_col_count_by_yr.T
    
        years_available = single_cols_to_select_df['year'].unique()
    
        for year in years_available:
            non_nan_count = len(single_cols_to_select_df[single_cols_to_select_df['year'] == year])
            single_col_count_by_yr[(year)] = single_col_count_by_yr[(year)].apply(lambda x: x/non_nan_count)


        # Increase the figure size
        fig, ax = plt.subplots(figsize=(20, 12))
    
        # Plot the data as a grouped bar chart
        single_col_count_by_yr.plot(kind='bar', stacked=False, ax=ax)
        plt.xlabel(f'{i}')
        plt.ylabel('% of respondents')
        plt.title(f'% of {i} by Year')

        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
        st.pyplot(fig)

        # (NEW) ====================
        single_col_count_by_yr['Average'] = single_col_count_by_yr.mean(axis=1) # NEW ======================
        single_col_count_by_yr = single_col_count_by_yr.sort_values(by='Average', ascending=False) # NEW ======================
        # Increase the figure size
        # Increase the figure size
        fig, ax = plt.subplots(figsize=(20, 12))
        plt.rcParams.update({'font.size': 14})  # Set the default font size for the figure

        # Plot the data as a bar chart
        single_col_count_by_yr['Average'].plot(kind='bar', ax=ax)
        plt.xlabel(f'{i}', fontsize=16)  # Set font size for x-axis label
        plt.ylabel('Average % of respondents', fontsize=16)  # Set font size for y-axis label
        plt.title(f'Average % of {i}', fontsize=18)  # Set font size for title

        # Rename the x-axis labels to show only the feature name
        new_labels = [label[1] for label in single_col_count_by_yr.index]
        ax.set_xticklabels(new_labels, rotation=45, horizontalalignment='right', fontsize=12)  # Set font size for x-axis tick labels

        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)  # Set font size for annotations

        st.pyplot(fig)

        # Extract the desired columns for the table after dropping level 0 index
        table_data = single_col_count_by_yr[['Average']].reset_index()

        # Display the table with 'new_labels' and 'Average' columns
        st.write('Average Values:')
        st.table(table_data)