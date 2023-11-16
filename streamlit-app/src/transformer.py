from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
import streamlit as st

class BinaryTransformer:

    def set_binary_cols(self, df):
        binary_cols = []
        for col in df.columns:
            if len(df[col].unique()) == 2:
                binary_cols.append(col)
        
        self.binary_cols = binary_cols

        return

    def fit(self, X, y = None, **fit_params):
        self.set_binary_cols(X)
        # print("before label encoder")
        self.le = {}
        for col in self.binary_cols:
            self.le[col] = LabelEncoder()
            self.le[col].fit(X[col])
        
        
        return
    
    def transform(self, X, y = None, **fit_params):
        res = []
        for col in self.binary_cols:
            res.append(self.le[col].transform(X[col]))
        # print(np.shape(np.array(res).T))
        return np.array(res).T
    

class MultiLabelTransformer:

    def set_non_binary_cols(self, df):
        non_binary_cols = []
        for col in df.columns:
            if len(df[col].unique()) > 2:
                non_binary_cols.append(col)
        
        non_binary_cols.remove("Location")

        self.non_binary_cols = non_binary_cols
        return
    
    def fit(self, X, y = None, **fit_params):
        self.set_non_binary_cols(X)

        self.ohe = OneHotEncoder()
        self.ohe.fit(X[self.non_binary_cols])
        
        # print(self.ohe.categories_)
        return
    
    def transform(self, X, y = None, **fit_params):
        X = self.ohe.transform(X[self.non_binary_cols])
        # print(np.shape(X))
        return X.toarray()

    
@st.cache_data
def FeatureTransformer(df):
    print("FeatureTransformer called")

    featureTransformer = FeatureUnion([
        ('binary_processing', Pipeline([('bnr', BinaryTransformer())])),
        ('mlbl', Pipeline([('mlbl', MultiLabelTransformer())])),
    ])

    featureTransformer.fit(df)
    # transformed_courses = featureTransformer.transform(datacomb_non_processed)

    return featureTransformer