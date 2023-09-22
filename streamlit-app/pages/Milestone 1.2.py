    
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
df = df[df['Country'].isin(sea_countries)]
df.Country.unique()
# Streamlit app
st.title('')

# Sidebar filters
st.sidebar.header('Select Columns to Display')
selected_column = st.sidebar.selectbox('Select Columns',['Age', 'Gender','Programming language'])

# Create a plot in the main section
st.header('Filtered Data and Plot')

if len(selected_column) > 0:
    count_data = pd.DataFrame(df.groupby([selected_column,'Year'])[selected_column].value_counts())
    count_data.reset_index(inplace=True)


# Create a plot using Matplotlib or any other plotting library
# Example using Matplotlib:
if not count_data.empty:
    fig, ax = plt.subplots(1,1)

    # plt.figure(figsize=(10, 6))  # Set the figure size (optional)
    sns.barplot(ax=ax, x=selected_column, y="count", hue="Year", data=count_data)

    # Set labels and title (customize as needed)
    plt.xlabel("Age Group")
    plt.xticks(rotation=45)
    plt.ylabel("Value")

    # Show the plot
    st.pyplot(fig)
else:
    st.warning('No data to plot.')
