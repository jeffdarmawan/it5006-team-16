from pprint import pprint
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# local imports
from src.transformer import FeatureTransformer

def print_all_columns(a):
    a = list(a)
    a.sort()
    return a


# 1. Load data
@st.cache_data
def getData(csvfile):
    """get the csv data"""
    data = pd.read_csv(csvfile)
    return data
datacomb_new = getData('data_allthreeyears_combined_new1.csv')

# 2. Data cleaning
# 2.1. Drop irrelevant columns
cols_to_drop = ['year', 'Job_No.OfDSTeamMember', 'Job_EmployerUsingML?','Money Spent on ML/Cloud Computing','Times used TPU']
datacomb_new = datacomb_new.drop(cols_to_drop, axis = 1)

# 2.2. Drop rows with empty job title and students
datacomb_new = datacomb_new.dropna(subset = ['Job_title - Selected Choice']) # drop rows with empty job title
datacomb_new = datacomb_new[datacomb_new['Job_title - Selected Choice'] != 'Student']# drop rows with student as job title
datacomb_new = datacomb_new[datacomb_new['Job_title - Selected Choice'] != 'Other']# drop rows with other as job title

# 2.3. Merge redundant job title
job_title_dict = {
    'Data Analyst (Business, Marketing, Financial, Quantitative, etc)': 'Data Analyst',
    'Product Manager': 'Product/Project/Program Manager',
    'Product/Project Manager': 'Product/Project/Program Manager',
    'Program/Project Manager':'Product/Project/Program Manager',
    'Machine Learning Engineer':'Machine Learning/ MLops Engineer'}

datacomb_new = datacomb_new.replace({'Job_title - Selected Choice': job_title_dict})
Job_title = datacomb_new['Job_title - Selected Choice']

# remove job title from df before encoding
cols_to_drop = ['Job_title - Selected Choice']
datacomb_new_wo_Jtitle = datacomb_new.drop(cols_to_drop, axis = 1)

# strip whitespace
datacomb_new_wo_Jtitle = datacomb_new_wo_Jtitle.map(lambda x: x.strip() if isinstance(x, str) else x)

# 3. Preprocessing - using transformers
unique_counts = datacomb_new.nunique(dropna=False)
binary_cols = unique_counts[unique_counts <= 2].index.tolist()
non_binary_cols = unique_counts[unique_counts > 2].index.tolist()


feature_transformer = FeatureTransformer(datacomb_new_wo_Jtitle)
encoded_df = feature_transformer.transform(datacomb_new_wo_Jtitle)

prefinal_columns = datacomb_new_wo_Jtitle.columns

# 4. Train test split
rng = np.random.RandomState(seed=321)
X_train, X_test, y_train, y_test = train_test_split( encoded_df, Job_title , test_size=0.20, random_state= rng)


# 5. Random Forest model building______________________________________
@st.cache_data
def getRandomForestModel(X_train,y_train):
    """Initialise a model and fit the training data set
       Return the fitted model which is ready to be used for prediction
    """
    best_param = {'bootstrap': False,
        'max_depth': 55,
        'max_features': 'sqrt',
        'min_samples_leaf': 2,
        'min_samples_split': 3,
        'n_estimators': 1100}
    model = RandomForestClassifier(**best_param)
    model.fit( X_train, y_train )
    return model

rnd_clf = getRandomForestModel(X_train=X_train, y_train=y_train  )

# ______________________________________Creating Questionaire ______________________________________

from src.x_testData import Datalist
data = Datalist.DATA.value

questions,answers = [],[]
for line in data:
    words = line.split(" ")
    if "-" in words:
        idx = [index for index, value in enumerate(words) if value == "-"][-1] # get last index of -
        questions.append(" ".join(words[:idx]))
        
        answers.append(" ".join(words[idx+1:]))
    else:
        questions.append(line)
        answers.append(1) # we need to update this
questions_answers = {}
# Iterate through the lists and populate the dictionary
for key, value in zip(questions, answers):
    if key in questions_answers:
        questions_answers[key].append(value)
    else:
        questions_answers[key] = [value]

multiSelectQns = Datalist.MULTI.value
singleSelectQns = Datalist.SINGLE.value
##############

#@st.cache_data
def get_recommendations(model, userInputs):
    """return the model prediction based on user input"""
    return model.predict(userInputs)

headers = list(datacomb_new_wo_Jtitle.columns) # used to encoding user input


def encodeUserInput(userinput):
    '''
        encode the user data to be passed into the random forest model
    '''
    single, multi = userinput # unpack the userinput

    ans = single.copy()
    for q, a in multi.items():
        if len(a) > 0:
            for indv_a in a:
                ans[q + " - " + indv_a] = indv_a

    ans_df = pd.DataFrame(ans, index=[0], columns=prefinal_columns)
    # transform the user input
    ans_df = feature_transformer.transform(ans_df)

    return ans_df


# load question map
qmap_df = pd.read_excel("Question theme to question mapping.xlsx", sheet_name = 'Question_Mapping')
qmap_dict = dict(qmap_df.to_numpy())


# _________________________ Streamlit UI ____________________________
st.title("Job Recommender System")


"""A function that create a list of questions and store the user input value"""
singleSelection = {}
multiList = {}
with st.form("questionnaire"):
    for question, answer_options in questions_answers.items():
        if question in singleSelectQns:   
            selected_answer = st.selectbox(qmap_dict[question], answer_options)
            singleSelection[question] = selected_answer 
        elif question in multiSelectQns:
            selected_answer = st.multiselect(qmap_dict[question], answer_options)
            multiList[question] = selected_answer
            
        st.write("---")  # Add a separator between questions
    
    if st.form_submit_button("Get Recommendations"):
        predictedJobs = get_recommendations(model=rnd_clf, userInputs=encodeUserInput((singleSelection, multiList)))
        st.write("Recommended Jobs based on your profile: ", predictedJobs[0])
