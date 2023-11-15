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

# 2.3. Merge redundant job title
job_title_dict = {
    'Data Analyst (Business, Marketing, Financial, Quantitative, etc)': 'Data Analyst',
    'Product Manager': 'Product/Project/Program Manager',
    'Product/Project Manager': 'Product/Project/Program Manager',
    'Program/Project Manager':'Product/Project/Program Manager',
    'Machine Learning Engineer':'Machine Learning/ MLops Engineer'}

datacomb_new = datacomb_new.replace({'Job_title - Selected Choice': job_title_dict})
Job_title = datacomb_new['Job_title - Selected Choice']

# 3. Preprocessing
# 3.1. Label Binary Columns to 0 and 1
unique_counts = datacomb_new.nunique(dropna=False)
binary_cols = unique_counts[unique_counts <= 2].index.tolist()
non_binary_cols = unique_counts[unique_counts > 2].index.tolist()

datacomb_new[binary_cols] = np.where(datacomb_new[binary_cols].isna(), 0, 1)


# 3.2. One-Hot Label Encoding

# remove job title from df before encoding
cols_to_drop = ['Job_title - Selected Choice']
datacomb_new_wo_Jtitle = datacomb_new.drop(cols_to_drop, axis = 1)
filtered_non_binary_cols = [item for item in non_binary_cols if item not in cols_to_drop]

prefinal_columns = datacomb_new_wo_Jtitle.columns
encoded_df = pd.get_dummies(datacomb_new_wo_Jtitle, columns = filtered_non_binary_cols)
encoded_df.drop('Age_70+', axis = 1, inplace = True) # to remove multi-colinearity


# Train test split
rng = np.random.RandomState(seed=321)
X_train, X_test, y_train, y_test = train_test_split( encoded_df, Job_title , test_size=0.20, random_state= rng)
# Random Forest model building______________________________________




@st.cache_data
def getRandomForestModel(n_estimators,max_leaf_nodes,n_jobs,X_train,y_train):
    
    # Open a file in write mode ('w')
    with open('x_train_cols.txt', 'w') as file:
        # print(type(X_train.columns))
        print(*print_all_columns(X_train.columns), file=file, sep="\n")

    """Initialise a model and fit the training data set
       Return the fitted model which is ready to be used for prediction
    """
    model = RandomForestClassifier( n_estimators=100, max_leaf_nodes=15, n_jobs=-1 )
    model.fit( X_train, y_train )
    return model

rnd_clf = getRandomForestModel( n_estimators=100, max_leaf_nodes=15, n_jobs=-1, X_train=X_train, y_train=y_train  )

# ______________________________________Training results______________________________________
# @st.cache_data
# def evaluate(test, pred):
#     print(classification_report( test, pred ))
#     # Calculate precision
#     precision = precision_score(test, pred, average='micro')
    
#     # Calculate recall
#     recall = recall_score(test, pred, average='micro')
    
#     print("Precision: ", precision)
#     print("Recall: ", recall)

# evaluate( y_test, y_pred_rf )
# ______________________________________Creating Questionaire ______________________________________

from pages.x_testData import Datalist
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
    # return model.predict(userInputs)

headers = list(datacomb_new_wo_Jtitle.columns) # used to encoding user input

#@st.cache_data
def encodeUserInput(userinput, headers=headers, columns=filtered_non_binary_cols):
    '''
        encode the user data to be passed into the random forest model
    '''
    single, multi = userinput #unpack the userinput
    # print("single:")
    # pprint(single)
    # pprint(single['Learning platforms tried - How well known are the platforms (platforms with good marketing)'])
    # print("multi:")
    # pprint(multi)
    # for col in headers.notin(columns):
    #     print(col)
    print(type(multi))
    ans = single.copy()
    for q, a in multi.items():
        if len(a) > 0:
            for indv_a in a:
                ans[q + " - " + indv_a] = indv_a
        print(q)
        print(a)


    print(ans)

    ans_df = pd.DataFrame(ans, index=[0], columns=prefinal_columns)

    print(ans_df)
    # === below might not be needed ===
    currentCols = list(single.keys() ) + list(multi.keys())
    emptyCols = list(set(headers) - set(currentCols)) # cols that are not present based on the user selected ans, will be set to 0 as they are the binary cols
    #overallAns = list(single.values()) + list(multi.values())
    overallAns = list(single.values()) + [1 for i in list(multi.values())]
    #overallAns =[1 for i in compiledAns]
    for i in emptyCols:
        overallAns.append(0) # set these feature as 0 
    overallHeaders = currentCols + emptyCols
    # print("Overall headers: ", overallHeaders)
    # print("Overall ans: ", overallAns)
    overallDF = pd.DataFrame([overallAns], columns=overallHeaders) 
    # Open a file in write mode ('w')
    with open('usercols.txt', 'w') as file:
        print(*print_all_columns(overallDF.columns),file=file, sep="\n")
    # print(f"DEBUGGGGG: len of overallheaders : {len(overallHeaders)}   ,len of overallAns : {len(overallAns)} ,len of columns : {len(columns)}       ")
    encoded_df = pd.get_dummies(overallDF, columns)
    return encoded_df


def create_questionnaire(questions_answers):
    """A function that create a list of questions and store the user input value"""
    singleSelection = {}
    multiList = {}
    
    for question, answer_options in questions_answers.items():
        if question in singleSelectQns:
            selected_answer = st.selectbox(question, answer_options)
            singleSelection[question] = selected_answer 
            #answerList.append(selected_answer) # for single select, do not append the question as it does not have multi cols
            # print("Answer is :",selected_answer)
            #answerList.append( "".join(question) + " - "+"".join(selected_answer))
        elif question in multiSelectQns:
            selected_answer = st.multiselect(question, answer_options)
            # print("Multi selected ans is :", selected_answer)
            multiList[question] = selected_answer
            #multiList.append(selected_answer)
            # for ans in selected_answer:
            #     multiList.append( "".join(question) + " - " + "".join(ans)) # the selection will include the 

        st.write(f"You selected for '{question}': {selected_answer}")
        st.write("---")  # Add a separator between questions

    return singleSelection, multiList

# Call the function to create the questionnaire
userinput = create_questionnaire(questions_answers)
# print("Userinput: ", userinputs)
# print("len of userinputs: ",len(userinputs))


# _________________________ Streamlit UI ____________________________
st.title("Job Recommender System")

if st.button("Get Recommendations"):
    get_recommendations(model=rnd_clf, userInputs=encodeUserInput(userinput))
    # predictedJobs = 
    st.write("Recommended Jobs based on your profile:")
    # st.write(predictedJobs)