
# Decision tree model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

#__________________________________________________pre-processing__________________________________________________

datacomb_new = pd.read_csv('data_allthreeyears_combined_new1.csv')
categorical_columns = datacomb_new.select_dtypes(include=['object']).columns.tolist()
datacomb_new = datacomb_new.drop('year', axis = 1) # drop year columns
unique_counts = datacomb_new.nunique(dropna=False)

binary_cols = unique_counts[unique_counts <= 2].index.tolist()
non_binary_cols = unique_counts[unique_counts > 2].index.tolist()

datacomb_new[binary_cols] = np.where((datacomb_new[binary_cols] != 0) & (~datacomb_new[binary_cols].isna()), 1, 0)
datacomb_new = datacomb_new.dropna(subset = ['Job_title - Selected Choice'])

Job_title = datacomb_new.pop('Job_title - Selected Choice')
datacomb_new.insert(len(datacomb_new.columns), 'Job_title - Selected Choice', Job_title)
cols_to_drop = ['Job_No.OfDSTeamMember', 'Job_EmployerUsingML?','Money Spent on ML/Cloud Computing','Times used TPU', 'Job_title - Selected Choice']
datacomb_new.drop(cols_to_drop, axis = 1, inplace = True)

filtered_non_binary_cols = [item for item in non_binary_cols if item not in cols_to_drop]
filtered_non_binary_cols

datacomb_new_wo_Jtitle = datacomb_new


# Encoding categorical features for X
encoded_df = pd.get_dummies(datacomb_new_wo_Jtitle, columns = filtered_non_binary_cols)

encoded_df.drop('Age_70+', axis = 1, inplace = True) # to remove multi-colinearity
encoded_array = encoded_df.to_numpy()



# Encoding labels in Y
y_encoded = label_encoder.fit_transform(pd.DataFrame(Job_title))


#__________________________________________________running__________________________________________________
# Step 1: X-Y dataset

X = encoded_df
y = Job_title

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 3: Train a Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = decision_tree_model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for more detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))