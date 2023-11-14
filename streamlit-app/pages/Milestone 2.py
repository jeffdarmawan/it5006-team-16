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

# import the csv 
@st.cache_data
def getData(csvfile):
    """get the csv data"""
    data = pd.read_csv(csvfile)
    return data
datacomb_new = getData('data_allthreeyears_combined_new1.csv')
# ___________________________________________ Data Pre-processing _________________________________________________________________ 
categorical_columns = datacomb_new.select_dtypes(include=['object']).columns.tolist()
datacomb_new = datacomb_new.drop('year', axis = 1) # drop the 'year' column
unique_counts = datacomb_new.nunique(dropna=False)
binary_cols = unique_counts[unique_counts <= 2].index.tolist()
non_binary_cols = unique_counts[unique_counts > 2].index.tolist()


# ___________________________________________ Label Binary Columns to 0 and 1 _________________________________________________________________ 
datacomb_new[binary_cols] = np.where((datacomb_new[binary_cols] != 0) & (~datacomb_new[binary_cols].isna()), 1, 0)

datacomb_new = datacomb_new.dropna(subset = ['Job_title - Selected Choice']) # drop rows with empty job title

datacomb_new = datacomb_new[datacomb_new['Job_title - Selected Choice'] != 'Student']# drop rows with student as job title
Job_title = datacomb_new.pop('Job_title - Selected Choice')
datacomb_new.insert(len(datacomb_new.columns), 'Job_title - Selected Choice', Job_title)



# ______________________________________Dropping cols we think is not associated to the job title___________________________________________________________ 
job_title_dict = {
    'Data Analyst (Business, Marketing, Financial, Quantitative, etc)': 'Data Analyst',
    'Product Manager': 'Product/Project/Program Manager',
    'Product/Project Manager': 'Product/Project/Program Manager',
    'Program/Project Manager':'Product/Project/Program Manager',
    'Machine Learning Engineer':'Machine Learning/ MLops Engineer'}

def replace_text(cell_value, replacements):
    if cell_value is not None and not pd.isna(cell_value):
        # Check if the cell_value is a float, and if so, convert it to a string.
        if isinstance(cell_value, float):
            cell_value = str(cell_value)
        cell_value = replacements.get(cell_value,cell_value)
    return cell_value

datacomb_new['Job_title - Selected Choice'] = datacomb_new['Job_title - Selected Choice'].apply(replace_text, replacements=job_title_dict)
Job_title = datacomb_new.pop('Job_title - Selected Choice')
datacomb_new.insert(len(datacomb_new.columns), 'Job_title - Selected Choice', Job_title)

cols_to_drop = ['Job_No.OfDSTeamMember', 'Job_EmployerUsingML?','Money Spent on ML/Cloud Computing','Times used TPU', 'Job_title - Selected Choice']
datacomb_new_wo_Jtitle = datacomb_new.drop(cols_to_drop, axis = 1)
filtered_non_binary_cols = [item for item in non_binary_cols if item not in cols_to_drop]


encoded_df = pd.get_dummies(datacomb_new_wo_Jtitle, columns = filtered_non_binary_cols)
encoded_df.drop('Age_70+', axis = 1, inplace = True) # to remove multi-colinearity


# ______________________________________Random Forest model building______________________________________


rng = np.random.RandomState(seed=321)
X_train, X_test, y_train, y_test = train_test_split( encoded_df, Job_title , test_size=0.20, random_state= rng)

@st.cache_data
def getRandomForestModel(n_estimators,max_leaf_nodes,n_jobs,X_train,y_train):
    """Initialise a model and fit the training data set
       Return the fitted model which is ready to be used for prediction
    """
    model = RandomForestClassifier( n_estimators=100, max_leaf_nodes=15, n_jobs=-1 )
    model.fit( X_train, y_train )
    return model

rnd_clf = getRandomForestModel( n_estimators=100, max_leaf_nodes=15, n_jobs=-1, X_train=X_train, y_train=y_train  )
#rnd_clf = RandomForestClassifier( n_estimators=100, max_leaf_nodes=15, n_jobs=-1 )
#rnd_clf.fit( X_train, y_train )
y_pred_rf = rnd_clf.predict( X_test )
#print( classification_report( y_test, y_pred_rf ))
print("Xtest is",X_test)

# ______________________________________Hyper parameter tuning______________________________________


# ______________________________________Training results______________________________________
@st.cache_data
def evaluate(test, pred):
    print(classification_report( test, pred ))
    # Calculate precision
    precision = precision_score(test, pred, average='micro')
    
    # Calculate recall
    recall = recall_score(test, pred, average='micro')
    
    print("Precision: ", precision)
    print("Recall: ", recall)

evaluate( y_test, y_pred_rf )
# ______________________________________Creating Questionaire ______________________________________

from x_testData import Datalist
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

@st.cache_data
def get_recommendations(model, userInputs):
    """return the model prediction based on user input"""
    return model.predict(userInputs)

headers = list(datacomb_new_wo_Jtitle.columns) # used to encoding user input

@st.cache_data
def encodeUserInput(userinput, headers=headers, columns=filtered_non_binary_cols):
    """encode the user data to be passed into the random forest model """
    single, multi = userinput #unpack the userinput
    currentCols = list(single.keys() ) + list(multi.keys())
    emptyCols = list(set(headers) - set(currentCols)) # cols that are not present based on the user selected ans, will be set to 0 as they are the binary cols
    #overallAns = list(single.values()) + list(multi.values())
    compiledAns = list(single.values()) + list(multi.values())
    overallAns =[1 for i in compiledAns]
    for i in emptyCols:
        overallAns.append(0) # set these feature as 0 
    overallHeaders = currentCols + emptyCols
    print("Overall headers: ", overallHeaders)
    print("Overall ans: ", overallAns)
    overallDF = pd.DataFrame([overallAns], columns=overallHeaders) 
    print(f"DEBUGGGGG: len of overallheaders : {len(overallHeaders)}   ,len of overallAns : {len(overallAns)} ,len of columns : {len(columns)}       ")
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
            print("Answer is :",selected_answer)
            #answerList.append( "".join(question) + " - "+"".join(selected_answer))
        elif question in multiSelectQns:
            selected_answer = st.multiselect(question, answer_options)
            print("Multi selected ans is :", selected_answer)
            multiList[question] = multiSelectQns
            #multiList.append(selected_answer)
            # for ans in selected_answer:
            #     multiList.append( "".join(question) + " - " + "".join(ans)) # the selection will include the 

        st.write(f"You selected for '{question}': {selected_answer}")
        st.write("---")  # Add a separator between questions

    return singleSelection,multiList

# Call the function to create the questionnaire
userinputs = create_questionnaire(questions_answers)
print("Userinput: ", userinputs)
print("len of userinputs: ",len(userinputs))


# _________________________ Streamlit UI ____________________________
st.title("Job Recommender System")

if st.button("Get Recommendations"):
    predictedJobs = get_recommendations(model=rnd_clf, userInputs=encodeUserInput(userinputs))
    st.write("Recommended Jobs based on your profile:")
    st.write(predictedJobs)