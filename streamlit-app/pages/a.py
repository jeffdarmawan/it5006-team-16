    
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

# Create a sample DataFrame
df = pd.read_csv('clean_dataset.csv.zip')
df.drop(columns=['Unnamed: 0'], inplace=True)
# print(df.groupby(['Age'])['Age'].value_counts())

# Streamlit app
st.title('Column Selection App')

# Sidebar filters
st.sidebar.header('Select Columns to Display')
selected_column = st.sidebar.selectbox('Select Columns', df.columns)

print("selected_column", selected_column)
print(type(selected_column))
# Create a plot in the main section
st.header('Filtered Data and Plot')

if len(selected_column) > 0:
    count_data = pd.DataFrame(df.groupby([selected_column,'Year'])[selected_column].value_counts())
    count_data.reset_index(inplace=True)
    print(count_data.head())
    print("printing count_data")
    print(count_data.head())


# Create a plot using Matplotlib or any other plotting library
# Example using Matplotlib:
if not count_data.empty:
    fig, ax = plt.subplots()

    # st.bar_chart(data=count_data, x=selected_column, y='count')
    chart = alt.Chart(count_data).mark_bar().encode(
        x=selected_column,
        y='count',
        color='Year:N',
        # column='Year'  # Separate bars by year
    ).properties(
        width=600,
        height=400
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning('No data to plot.')
