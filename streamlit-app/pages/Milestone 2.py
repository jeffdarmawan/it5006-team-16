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
#print(datacomb_new_wo_Jtitle.columns)
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

def create_questionnaire(questions_answers):
    for question, answer_options in questions_answers.items():
        selected_answer = st.selectbox(question, answer_options)
        st.write(f"You selected for '{question}': {selected_answer}")
        st.write("---")  # Add a separator between questions

# Define your questions and answers
questions_answers = {
    "What is your name?": ["Alice", "Bob", "Charlie"],
    "How old are you?": ["Under 18", "18-30", "31-50", "Over 50"],
    "What is your favorite color?": ["Red", "Blue", "Green", "Other"],
    # Add more questions and answers as needed
}

# Call the function to create the questionnaire
create_questionnaire(questions_answers)

def user_input():
    """return the list of attribute user has provided to pass into the model for prediction"""
    return

@st.cache
def get_recommendations(model, userInputs):
    """return the model prediction based on user input"""
    return model.predict(userInputs)
    



# _________________________ Streamlit UI ____________________________
st.title("Job Recommender System")

if st.button("Get Recommendations"):
    recommendations = get_recommendations(model = rnd_clf, userInputs=user_input())
    st.write("Recommended Jobs based on your profile:")
    st.write(recommendations)