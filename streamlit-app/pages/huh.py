import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random

# Create dummy data
data = {
    'age_group': ['0-18', '19-35', '36-50', '51+'] * 3,
    'year': [2019] * 4 + [2020] * 4 + [2021] * 4,
    'value': [random.randint(1, 100) for _ in range(12)]
}

# Create a DataFrame from the dummy data
df = pd.DataFrame(data)

# Streamlit app
st.title("Grouped Bar Plot by Age Group and Year")

# Sidebar for customization (optional)
# You can add widgets to allow users to customize the plot if needed

# Create a grouped bar plot
plt.figure(figsize=(10, 6))  # Set the figure size (optional)
sns.barplot(x="age_group", y="value", hue="year", data=df)

# Set labels and title (customize as needed)
plt.xlabel("Age Group")
plt.ylabel("Value")

# Show the plot
st.pyplot()

# Optionally, you can add other content to your Streamlit app below the plot
