    
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame
#datacomb = pd.read_csv('clean_dataset.csv.zip')
datacomb = pd.read_csv("data_allthreeyears_combined.csv")


# creating multiple selections for the user to select
multi_select_cols = set()
print("df_salary_exp_heatmap_data new cols", df_salary_exp_heatmap_data.columns)

for col in df_salary_exp_heatmap_data.columns:
    if " - " in col:
        multi_select_qn = col.split(" - ")[0]
        multi_select_cols.add(multi_select_qn)

print("multi_select_cols is :", multi_select_cols)




# Streamlit app
st.title('helpppp')
# Sidebar filters
st.sidebar.header('Select Columns to Display')

selected_column = st.sidebar.multiselect('Select Columns',multi_select_cols)

start_year, end_year = st.sidebar.select_slider(
    'Select period',
    options=[2020, 2021, 2022],
    value=(2020, 2022))

# Create a plot in the main section
st.header('Filtered Data and Plot')


selected_years = np.arange(start_year, end_year, 1)
selected_years =  np.append(selected_years, end_year)

def generate_heatmap(data):
    """Function to generate heatmap in streamlit"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu")
    st.pyplot()


# Data processing based on the columns users chose
if len(selected_column) > 0:
    selected_df = datacomb[datacomb['Year'].isin(selected_years)]
    # datacomb.groupby([selected_column,'Year'])[selected_column].value_counts()
    count_data = pd.DataFrame(selected_df.groupby([selected_column,'Year'])[selected_column].value_counts())
    count_by_year = pd.DataFrame(count_data.groupby(['Year']).sum())
    count_by_year.rename(columns={'count': 'total'}, inplace=True)
    count_data.reset_index(inplace=True)
    count_data = count_data.merge( count_by_year, left_on='Year', right_on='Year', how='left')
    count_data['percentage'] = count_data['count'] / count_data['total'] * 100
    print("**************************")
    print(count_data)



# Create a plot using seaborn
if not count_data.empty:
    fig, ax = plt.subplots(1,1)

    sns.barplot(ax=ax, x=selected_column, y="percentage", hue="Year", data=count_data)

    # Set labels and title (customize as needed)
    plt.xlabel("Age Group")
    plt.xticks(rotation=45)
    plt.ylabel("Percentage")

    # Show the plot
    st.pyplot(fig)
else:
    st.warning('No data to plot.')
