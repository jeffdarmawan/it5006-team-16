import pandas as pd

# Sample data with unique values for all columns except 'Job_Salary'
data_except_salary = {
    'Age': [25, 30, 35, 40],
    'Education level_attainedOrGGtoAttain': ['High School', 'Bachelor', 'Master', 'Ph.D.'],
    'Coding Experience (in years)': [2, 5, 10, 15],
    'Years in ML': [1, 3, 5, 7],
    'Job_No.OfDSTeamMember': [1, 5, 10, 20],
    'Money Spent on ML/Cloud Computing': [100, 500, 1000, 2000],
    'Times used TPU': [0, 1, 5, 10]
}

# Data for the 'Job_Salary' column with the specified values
data_salary = {
    'Job_Salary': [50000, 70000, 90000, 120000, 150000]
}

# Create DataFrames from the separate data dictionaries
df_except_salary = pd.DataFrame(data_except_salary)
df_salary = pd.DataFrame(data_salary)

# Concatenate the DataFrames horizontally (axis=1) using column names
df = pd.concat([df_except_salary, df_salary], axis=1)

# Print the resulting DataFrame
print(df)
