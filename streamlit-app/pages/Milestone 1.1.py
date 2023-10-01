    
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame
df = pd.read_csv('clean_dataset.csv.zip')
df.drop(columns=['Unnamed: 0'], inplace=True)
# print(df.groupby(['Age'])['Age'].value_counts())
sea_countries = ["Brunei","Cambodia","East Timor","Indonesia","Laos","Malaysia","Myanmar","Philippines","Singapore","Thailand","Vietnam"]

# print(df['Job_Salary'].unique())
# df = df[df['Country'].isin(sea_countries)]

print(df[df['Country'].isin(sea_countries)].shape)
print(df[df['Gender'].isin(['Woman'])].shape)
df = df[df['Gender'].isin(['Man', 'Woman'])]

# Streamlit app
st.title('')

# Sidebar filters
st.sidebar.header('Select Columns to Display')
selected_column = st.sidebar.selectbox('Select Columns',['Age', 'Gender'])





# normalized = pd.DataFrame(pop_prog_lang_python)
# normalized['Total Respondents'] = [len(pop_prog_lang_2020),len(pop_prog_lang_2021),len(pop_prog_lang_2022)]
# normalized['Percentage - Python'] = normalized['Popular programming language - Python'] / normalized['Total Respondents'] * 100
# normalized


# Create a plot in the main section
st.header('Filtered Data and Plot')

if len(selected_column) > 0:
    # df.groupby([selected_column,'Year'])[selected_column].value_counts()
    count_data = pd.DataFrame(df.groupby([selected_column,'Year'])[selected_column].value_counts())
    count_by_year = pd.DataFrame(count_data.groupby(['Year']).sum())
    count_by_year.rename(columns={'count': 'total'}, inplace=True)
    count_data.reset_index(inplace=True)
    count_data = count_data.merge( count_by_year, left_on='Year', right_on='Year', how='left')
    count_data['percentage'] = count_data['count'] / count_data['total'] * 100

    # col:
    # - age
    # - year
    # - count 

# Create a plot using Matplotlib or any other plotting library
# Example using Matplotlib:
if not count_data.empty:
    fig, ax = plt.subplots(1,1)

    sns.barplot(ax=ax, x=selected_column, y="count", hue="Year", data=count_data)

    # Set labels and title (customize as needed)
    plt.xlabel("Age Group")
    plt.xticks(rotation=45)
    plt.ylabel("Percentage")

    # Show the plot
    st.pyplot(fig)
else:
    st.warning('No data to plot.')
