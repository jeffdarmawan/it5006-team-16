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



###############
data = [
'Age',
 'Coding Experience (in years)',
 'Education level_attainedOrGGtoAttain',
 'Gender - Selected Choice',
 'Job_JobScope - Analyze and understand data to influence product or business decisions',
 'Job_JobScope - Build and/or run a machine learning service that operationally improves my product or workflows',
 'Job_JobScope - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
 'Job_JobScope - Build prototypes to explore applying machine learning to new areas',
 'Job_JobScope - Do research that advances the state of the art of machine learning',
 'Job_JobScope - Experimentation and iteration to improve existing ML models',
 'Job_Salary',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - Cloud-certification programs (direct from AWS, Azure, GCP, or similar)',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - Coursera',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - DataCamp',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - Fast.ai',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - Kaggle Learn Courses',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - LinkedIn Learning',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - Udacity',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - Udemy',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - University Courses (resulting in a university degree)',
 'Learning platforms tried - How well known are the platforms (platforms with good marketing) - edX',
 'Popular BI tool brands - Alteryx',
 'Popular BI tool brands - Amazon QuickSight',
 'Popular BI tool brands - Domo',
 'Popular BI tool brands - Einstein Analytics',
 'Popular BI tool brands - Google Data Studio',
 'Popular BI tool brands - Looker',
 'Popular BI tool brands - Microsoft Azure Synapse',
 'Popular BI tool brands - Microsoft Power BI',
 'Popular BI tool brands - Qlik',
 'Popular BI tool brands - Qlik Sense',
 'Popular BI tool brands - SAP Analytics Cloud',
 'Popular BI tool brands - Salesforce',
 'Popular BI tool brands - Sisense',
 'Popular BI tool brands - TIBCO Spotfire',
 'Popular BI tool brands - Tableau',
 'Popular BI tool brands - Tableau CRM',
 'Popular BI tool brands - Thoughtspot',
 'Popular Cloud Computing Platform Brand - Alibaba Cloud',
 'Popular Cloud Computing Platform Brand - Amazon Web Services (AWS)',
 'Popular Cloud Computing Platform Brand - Google Cloud Platform (GCP)',
 'Popular Cloud Computing Platform Brand - Huawei Cloud',
 'Popular Cloud Computing Platform Brand - IBM Cloud / Red Hat',
 'Popular Cloud Computing Platform Brand - Microsoft Azure',
 'Popular Cloud Computing Platform Brand - Oracle Cloud',
 'Popular Cloud Computing Platform Brand - SAP Cloud',
 'Popular Cloud Computing Platform Brand - Salesforce Cloud',
 'Popular Cloud Computing Platform Brand - Tencent Cloud',
 'Popular Cloud Computing Platform Brand - VMware Cloud',
 'Popular Cloud Computing Product Brand - AWS Lambda',
 'Popular Cloud Computing Product Brand - Amazon EC2',
 'Popular Cloud Computing Product Brand - Amazon Elastic Compute Cloud (EC2)',
 'Popular Cloud Computing Product Brand - Amazon Elastic Container Service',
 'Popular Cloud Computing Product Brand - Azure Cloud Services',
 'Popular Cloud Computing Product Brand - Azure Functions',
 'Popular Cloud Computing Product Brand - Google Cloud App Engine',
 'Popular Cloud Computing Product Brand - Google Cloud Compute Engine',
 'Popular Cloud Computing Product Brand - Google Cloud Functions',
 'Popular Cloud Computing Product Brand - Google Cloud Run',
 'Popular Cloud Computing Product Brand - Microsoft Azure Container Instances',
 'Popular Cloud Computing Product Brand - Microsoft Azure Virtual Machines',
 'Popular Computer Vision Methods - General purpose image/video tools (PIL, cv2, skimage, etc)',
 'Popular Computer Vision Methods - Generative Networks (GAN, VAE, etc)',
 'Popular Computer Vision Methods - Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)',
 'Popular Computer Vision Methods - Image segmentation methods (U-Net, Mask R-CNN, etc)',
 'Popular Computer Vision Methods - Object detection methods (YOLOv3, RetinaNet, etc)',
 'Popular Computer Vision Methods - Object detection methods (YOLOv6, RetinaNet, etc)',
 'Popular Computer Vision Methods - Vision transformer networks (ViT, DeiT, BiT, BEiT, Swin, etc)',
 'Popular ML Algorithms - Autoencoder Networks (DAE, VAE, etc)',
 'Popular ML Algorithms - Bayesian Approaches',
 'Popular ML Algorithms - Convolutional Neural Networks',
 'Popular ML Algorithms - Decision Trees or Random Forests',
 'Popular ML Algorithms - Dense Neural Networks (MLPs, etc)',
 'Popular ML Algorithms - Evolutionary Approaches',
 'Popular ML Algorithms - Generative Adversarial Networks',
 'Popular ML Algorithms - Gradient Boosting Machines (xgboost, lightgbm, etc)',
 'Popular ML Algorithms - Graph Neural Networks',
 'Popular ML Algorithms - Linear or Logistic Regression',
 'Popular ML Algorithms - Recurrent Neural Networks',
 'Popular ML Algorithms - Transformer Networks (BERT, gpt-3, etc)',
 'Popular ML frameworks - Caret',
 'Popular ML frameworks - CatBoost',
 'Popular ML frameworks - Fast.ai',
 'Popular ML frameworks - H2O 3',
 'Popular ML frameworks - Huggingface',
 'Popular ML frameworks - JAX',
 'Popular ML frameworks - Keras',
 'Popular ML frameworks - LightGBM',
 'Popular ML frameworks - MXNet',
 'Popular ML frameworks - Prophet',
 'Popular ML frameworks - PyTorch',
 'Popular ML frameworks - PyTorch Lightning',
 'Popular ML frameworks - Scikit-learn',
 'Popular ML frameworks - TensorFlow',
 'Popular ML frameworks - Tidymodels',
 'Popular ML frameworks - Xgboost',
 'Popular ML product brand - Alteryx',
 'Popular ML product brand - Amazon Forecast',
 'Popular ML product brand - Amazon Rekognition',
 'Popular ML product brand - Amazon SageMaker',
 'Popular ML product brand - Azure Cognitive Services',
 'Popular ML product brand - Azure Machine Learning Studio',
 'Popular ML product brand - C3.ai',
 'Popular ML product brand - DataRobot',
 'Popular ML product brand - Databricks',
 'Popular ML product brand - Dataiku',
 'Popular ML product brand - Domino Data Lab',
 'Popular ML product brand - Google Cloud AI Platform / Google Cloud ML Engine',
 'Popular ML product brand - Google Cloud Natural Language',
 'Popular ML product brand - Google Cloud Vertex AI',
 'Popular ML product brand - Google Cloud Video AI',
 'Popular ML product brand - Google Cloud Vision AI',
 'Popular ML product brand - H2O AI Cloud',
 'Popular ML product brand - Rapidminer',
 'Popular NLP Methods - Contextualized embeddings (ELMo, CoVe)',
 'Popular NLP Methods - Encoder-decoder models (seq2seq, vanilla transformers)',
 'Popular NLP Methods - Encoder-decorder models (seq2seq, vanilla transformers)',
 'Popular NLP Methods - Transformer language models (GPT-3, BERT, XLnet, etc)',
 'Popular NLP Methods - Word embeddings/vectors (GLoVe, fastText, word2vec)',
 'Popular auto ML product brand - Amazon Sagemaker Autopilot',
 'Popular auto ML product brand - Auto-Keras',
 'Popular auto ML product brand - Auto-Sklearn',
 'Popular auto ML product brand - Auto_ml',
 'Popular auto ML product brand - Azure Automated Machine Learning',
 'Popular auto ML product brand - DataRobot AutoML',
 'Popular auto ML product brand - Databricks AutoML',
 'Popular auto ML product brand - Google Cloud AutoML',
 'Popular auto ML product brand - H20 Driverless AI',
 'Popular auto ML product brand - H2O Driverless AI',
 'Popular auto ML product brand - MLbox',
 'Popular auto ML product brand - Tpot',
 'Popular auto ML product brand - Xcessiv',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Amazon Athena',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Amazon Aurora',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Amazon DynamoDB',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Amazon RDS',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Amazon Redshift',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Google Cloud BigQuery',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Google Cloud BigTable',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Google Cloud Firestore',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Google Cloud SQL',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Google Cloud Spanner',
 'Popular data product brands used (Databases, Warehouses, Lakes) - IBM Db2',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Microsoft Access',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Microsoft Azure Cosmos DB',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Microsoft Azure Data Lake Storage',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Microsoft Azure SQL Database',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Microsoft SQL Server',
 'Popular data product brands used (Databases, Warehouses, Lakes) - MongoDB',
 'Popular data product brands used (Databases, Warehouses, Lakes) - MySQL',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Oracle Database',
 'Popular data product brands used (Databases, Warehouses, Lakes) - PostgreSQL',
 'Popular data product brands used (Databases, Warehouses, Lakes) - PostgresSQL',
 'Popular data product brands used (Databases, Warehouses, Lakes) - SQLite',
 'Popular data product brands used (Databases, Warehouses, Lakes) - Snowflake',
 'Popular hosted notebook products - Amazon EMR Notebooks',
 'Popular hosted notebook products - Amazon Sagemaker Studio',
 'Popular hosted notebook products - Amazon Sagemaker Studio Lab',
 'Popular hosted notebook products - Amazon Sagemaker Studio Notebooks',
 'Popular hosted notebook products - Azure Notebooks',
 'Popular hosted notebook products - Binder / JupyterHub',
 'Popular hosted notebook products - Code Ocean',
 'Popular hosted notebook products - Colab Notebooks',
 'Popular hosted notebook products - Databricks Collaborative Notebooks',
 'Popular hosted notebook products - Deepnote Notebooks',
 'Popular hosted notebook products - Google Cloud AI Platform Notebooks',
 'Popular hosted notebook products - Google Cloud Datalab',
 'Popular hosted notebook products - Google Cloud Datalab Notebooks',
 'Popular hosted notebook products - Google Cloud Notebooks (AI Platform / Vertex AI)',
 'Popular hosted notebook products - Google Cloud Vertex AI Workbench',
 'Popular hosted notebook products - Gradient Notebooks',
 'Popular hosted notebook products - Hex Workspaces',
 'Popular hosted notebook products - IBM Watson Studio',
 'Popular hosted notebook products - Kaggle Notebooks',
 'Popular hosted notebook products - Noteable Notebooks',
 'Popular hosted notebook products - Observable Notebooks',
 'Popular hosted notebook products - Paperspace / Gradient',
 'Popular hosted notebook products - Zeppelin / Zepl Notebooks',
 'Popular media sources for Data Science - Blogs (Towards Data Science, Analytics Vidhya, etc)',
 'Popular media sources for Data Science - Course Forums (forums.fast.ai, Coursera forums, etc)',
 "Popular media sources for Data Science - Email newsletters (Data Elixir, O'Reilly Data & AI, etc)",
 'Popular media sources for Data Science - Journal Publications (peer-reviewed journals, conference proceedings, etc)',
 'Popular media sources for Data Science - Kaggle (notebooks, forums, etc)',
 'Popular media sources for Data Science - Podcasts (Chai Time Data Science, Oâ€™Reilly Data Show, etc)',
 'Popular media sources for Data Science - Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)',
 'Popular media sources for Data Science - Reddit (r/machinelearning, etc)',
 'Popular media sources for Data Science - Slack Communities (ods.ai, kagglenoobs, etc)',
 'Popular media sources for Data Science - Twitter (data science influencers)',
 'Popular media sources for Data Science - YouTube (Kaggle YouTube, Cloud AI Adventures, etc)',
 'Popular programming language - Bash',
 'Popular programming language - C',
 'Popular programming language - C#',
 'Popular programming language - C++',
 'Popular programming language - Go',
 'Popular programming language - Java',
 'Popular programming language - Javascript',
 'Popular programming language - Julia',
 'Popular programming language - MATLAB',
 'Popular programming language - PHP',
 'Popular programming language - Python',
 'Popular programming language - R',
 'Popular programming language - SQL',
 'Popular programming language - Swift',
 'Popular tools to monitor ML/Experiments - Aporia',
 'Popular tools to monitor ML/Experiments - Arize',
 'Popular tools to monitor ML/Experiments - ClearML',
 'Popular tools to monitor ML/Experiments - Comet.ml',
 'Popular tools to monitor ML/Experiments - DVC',
 'Popular tools to monitor ML/Experiments - Domino Model Monitor',
 'Popular tools to monitor ML/Experiments - Evidently AI',
 'Popular tools to monitor ML/Experiments - Fiddler',
 'Popular tools to monitor ML/Experiments - Guild.ai',
 'Popular tools to monitor ML/Experiments - MLflow',
 'Popular tools to monitor ML/Experiments - Neptune.ai',
 'Popular tools to monitor ML/Experiments - Polyaxon',
 'Popular tools to monitor ML/Experiments - Sacred + Omniboard',
 'Popular tools to monitor ML/Experiments - TensorBoard',
 'Popular tools to monitor ML/Experiments - Trains',
 'Popular tools to monitor ML/Experiments - Weights & Biases',
 'Popular tools to monitor ML/Experiments - WhyLabs',
 'Years in ML']
########

questions,answers = [],[]
for line in data:
    words = line.split(" ")
    if "-" in words:
        idx = [index for index, value in enumerate(words) if value == "-"][-1] # get last index of -
        questions.append(" ".join(words[:idx]))
        
        answers.append(" ".join(words[idx+1:]))
    else:
        questions.append(line)
        answers.append("None")
questions_answers = {}
# Iterate through the lists and populate the dictionary
for key, value in zip(questions, answers):
    if key in questions_answers:
        questions_answers[key].append(value)
    else:
        questions_answers[key] = [value]

##############

def create_questionnaire(questions_answers):
    for question, answer_options in questions_answers.items():
        selected_answer = st.selectbox(question, answer_options)
        st.write(f"You selected for '{question}': {selected_answer}")
        st.write("---")  # Add a separator between questions


# Call the function to create the questionnaire
create_questionnaire(questions_answers)

def user_input():
    """return the list of attribute user has provided to pass into the model for prediction"""
    return

@st.cache
def get_recommendations(model, userInputs):
    """return the model prediction based on user input"""
    return model.predict(userInputs)
    
@st.cache
def encodeUserInput(userinput):
    """encode the user data to be passed into the random forest model """
    return 
# _________________________ Streamlit UI ____________________________
st.title("Job Recommender System")

if st.button("Get Recommendations"):
    #recommendations = get_recommendations(model = rnd_clf, userInputs=encodeUserInput(user_input())) 
    st.write("Recommended Jobs based on your profile:")
    st.write(recommendations)