import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


kaggle2020 = pd.read_csv( "https://raw.githubusercontent.com/jeffdarmawan/it5006-team-16/main/kaggle-survey-2021/kaggle_survey_2021_responses.csv" )
kaggle2021 = pd.read_csv( "https://raw.githubusercontent.com/jeffdarmawan/it5006-team-16/main/kaggle-survey-2021/kaggle_survey_2021_responses.csv" )
kaggle2022 = pd.read_csv( "https://raw.githubusercontent.com/jeffdarmawan/it5006-team-16/main/kaggle-survey-2021/kaggle_survey_2021_responses.csv" )

st.write(kaggle2020.head())

# replace columns name with the first role and remove the first row
kaggle2020.columns = kaggle2020.iloc[0]
kaggle2020 = kaggle2020.iloc[1:,:]
kaggle2021.columns = kaggle2021.iloc[0]
kaggle2021 = kaggle2021.iloc[1:,:]
kaggle2022.columns = kaggle2022.iloc[0]
kaggle2022 = kaggle2022.iloc[1:,:]


# remove NAN columns


xRows = [['What is your age (# years)?'], ['In which country do you currently reside?']]




st.subheader('Trends')

#if kaggle2020 is not None:
#    kaggle2020['What is your age (# years)?'] = pd.to_datetime(kaggle2020['Date'])
#    plt.plot(kaggle2020['Date'], kaggle2020['Value'])
#    plt.xlabel('Date')
#    plt.ylabel('Value')
#    st.pyplot()


#st.title('My Streamlit App')
#st.write('Hello, Streamlit!')


