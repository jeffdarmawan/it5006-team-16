# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########
# 2022 #
########
data_2022 = pd.read_excel("https://github.com/jeffdarmawan/it5006-team-16/raw/main/kaggle_survey_2022_2021_2020_responses_SB.xlsx", sheet_name= 'survey_2022')

data_2022.head()

row_to_check = 0

cols_to_drop = []

for column in data_2022.columns:
    # Check if the word is present in the cell at the specified row and column
    if data_2022.at[row_to_check, column] == "drop":
        cols_to_drop.append(column)

# Drop the selected columns
data2022 = data_2022.drop(columns=cols_to_drop)

# strip whitespace in column headers
data2022.columns = data2022.columns.str.strip()

questions = data2022.iloc[2]

columns = data2022.columns
columns = columns.str.replace(r'\d+', '', regex=True)
columns = columns.str.strip()

new_columns = []
i = 0
for dat in questions:
    new_name = columns[i]
    question_split = dat.split(' - ')

    if len(question_split) > 1:
        ans = question_split[-1].strip()
        new_name = new_name + ' - ' + ans
    new_columns.append(new_name)
    i += 1

data2022.columns = new_columns

data2022 = data2022.iloc[3:].reset_index(drop=True)

data2022['year'] = 2022

########
# 2021 #
########

data_2021 = pd.read_excel("https://github.com/jeffdarmawan/it5006-team-16/raw/main/kaggle_survey_2022_2021_2020_responses_SB.xlsx", sheet_name= 'survey_2021')

row_to_check = 0

cols_to_drop = []

for column in data_2021.columns:
    # Check if the word is present in the cell at the specified row and column
    if data_2021.at[row_to_check, column] == "drop":
        cols_to_drop.append(column)

# Drop the selected columns
data2021 = data_2021.drop(columns=cols_to_drop)

# strip whitespace in column headers
data2021.columns = data2021.columns.str.strip()

questions = data2021.iloc[2]

columns = data2021.columns
columns = columns.str.replace(r'\d+', '', regex=True)
columns = columns.str.strip()

new_columns = []
i = 0
for dat in questions:
    new_name = columns[i]
    question_split = dat.split(' - ')

    if len(question_split) > 1:
        ans = question_split[-1].strip()
        new_name = new_name + ' - ' + ans
    new_columns.append(new_name)
    i += 1


data2021.columns = new_columns

data2021 = data2021.iloc[3:].reset_index(drop=True)
# add year label
data2021['year'] = 2021

########
# 2020 #
########

data_2020 = pd.read_excel("https://github.com/jeffdarmawan/it5006-team-16/raw/main/kaggle_survey_2022_2021_2020_responses_SB.xlsx", sheet_name= 'survey_2020')

row_to_check = 0

cols_to_drop = []

for column in data_2020.columns:
    # Check if the word is present in the cell at the specified row and column
    if data_2020.at[row_to_check, column] == "drop":
        cols_to_drop.append(column)

# Drop the selected columns
data2020 = data_2020.drop(columns=cols_to_drop)

# strip whitespace in column headers
data2020.columns = data2020.columns.str.strip()

questions = data2020.iloc[2]

columns = data2020.columns
columns = columns.str.replace(r'\d+', '', regex=True)
columns = columns.str.strip()

new_columns = []
i = 0
for dat in questions:
    new_name = columns[i]
    question_split = dat.split(' - ')

    if len(question_split) > 1:
        ans = question_split[-1].strip()
        new_name = new_name + ' - ' + ans
    new_columns.append(new_name)
    i += 1


data2020.columns = new_columns

data2020.columns

data2020= data2020.iloc[3:].reset_index(drop=True)

data2020.head()

data2020['Times used TPU'].unique()

data2020['year'] = 2020

# Find columns that are unique to each DataFrame
columns_only_in_data2022 = set(data2022.columns) - set(data2021.columns) - set(data2020.columns)
columns_only_in_data2021 = set(data2021.columns) - set(data2022.columns) - set(data2020.columns)
columns_only_in_data2020 = set(data2020.columns) - set(data2022.columns) - set(data2021.columns)

# Print the results
print("Columns only in data2022:", columns_only_in_data2022)
print("Columns only in data2021:", columns_only_in_data2021)
print("Columns only in data2020:", columns_only_in_data2020)

data2022_cols = pd.DataFrame(data2022.columns)

datacomb = pd.concat([data2022, data2021, data2020], axis=0)


# writing the datacomb dataframe to excel
datacomb.to_csv('data_allthreeyears_combined.xlsx', index=False)

datacomb['Coding Experience (in years)'].unique()

for i in datacomb['Education level_attainedOrGGtoAttain'].unique():
    print(f'{i}: {type(i)}')

datacomb['Job_Salary'].unique()

"""# Cleaning up the 'Education level_attainedOrGGtoAttain' column"""

# there are some weird symbols in some cells

replacement_dict = {
    'Bachelor’s degree': 'bachelors',
    'Master’s degree': 'masters',
    'Some college/university study without earning a bachelor’s degree': 'college without bachelors',
    'Doctoral degree':'doctoral',
    'I prefer not to answer': None,
    'Professional doctorate':'doctorate',
    'No formal education past high school': 'high school and below',
    'Bachelorâ€™s degree':'bachelors',
    'Masterâ€™s degree': 'masters',
    'Some college/university study without earning a bachelorâ€™s degree': 'college without bachelors',
    'Professional degree': 'professional deg'}


def replace_text(cell_value, replacements):
    if cell_value is not None and not pd.isna(cell_value):
        # Check if the cell_value is a float, and if so, convert it to a string.
        if isinstance(cell_value, float):
            cell_value = str(cell_value)
        cell_value = replacements.get(cell_value,cell_value)
    return cell_value


# def replace_text(cell_value, replacements):
#     if not pd.isna(cell_value) and cell_value in replacements:
#         return replacements[cell_value]
#     return cell_value

datacomb['Education level_attainedOrGGtoAttain'] = datacomb['Education level_attainedOrGGtoAttain'].apply(replace_text, replacements=replacement_dict)

datacomb['Education level_attainedOrGGtoAttain'].unique()

# display all columns
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

# datacomb['Popular programming language3']

datafrm = pd.DataFrame(datacomb.columns)
datafrm



"""# Data Exploration"""

datacomb.columns

columns_to_include = ['year'] + [col for col in datacomb.columns if 'Popular programming language' in col]

columns_to_include

len(columns_to_include)

columns_to_include = ['year'] + [col for col in datacomb.columns if 'Popular programming language' in col]

pop_prog_lang = datacomb.loc[:, columns_to_include]
pop_prog_lang

# pop_prog_lang_cleaned = pop_prog_lang.dropna(subset=pop_prog_lang.columns.difference(['year']), how='all')
# pop_prog_lang_cleaned

pop_prog_lang_with_ans = pop_prog_lang.dropna(subset=pop_prog_lang.columns.difference(['year']),how='all')

pop_prog_lang_2022 = pop_prog_lang_with_ans[pop_prog_lang_with_ans['year']==2022]
pop_prog_lang_2021 = pop_prog_lang_with_ans[pop_prog_lang_with_ans['year']==2021]
pop_prog_lang_2020 = pop_prog_lang_with_ans[pop_prog_lang_with_ans['year']==2020]

# pop_prog_lang.groupby('year').count().sum(axis=1)

# pop_prog_lang.groupby('year').count()

pop_prog_lang_python = datacomb.groupby('year')['Popular programming language - Python'].count()


normalized = pd.DataFrame(pop_prog_lang_python)
normalized['Total Respondents'] = [len(pop_prog_lang_2020),len(pop_prog_lang_2021),len(pop_prog_lang_2022)]
normalized['Percentage - Python'] = normalized['Popular programming language - Python'] / normalized['Total Respondents'] * 100
normalized

pop_prog_lang_python_norm = pop_prog_lang_python.div(pop_prog_lang_cleaned.count().max())

pop_prog_lang_python_norm

pop_prog_lang_test = pd.read_csv('pop_programming_language_test.csv')

pop_prog_lang_test

