import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load your data
df = pd.read_csv('ai_job_impact.csv')

# Define features and target
features = ['Age', 'Years_Experience', 'Salary_Before_AI', 'Work_Hours_Per_Week', 'Job_Satisfaction', 'Education_Level', 'Industry']
target = 'Salary_After_AI'

X = df[features]
y = df[target]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Age', 'Years_Experience', 'Salary_Before_AI', 'Work_Hours_Per_Week', 'Job_Satisfaction']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Education_Level', 'Industry'])
    ])

# Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Success: model.pkl has been created!")