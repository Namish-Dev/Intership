
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib


CSV_PATH = "data/adult 3.csv"
df = pd.read_csv(CSV_PATH)

df['workclass'] = df['workclass'].replace('?', 'Others')
df['occupation'] = df['occupation'].replace('?', 'Others')

drop_cols = [
    'fnlwgt', 'marital-status', 'relationship', 'capital-loss', 'capital-gain',
    'educational-num', 'race', 'gender'
]
df.drop(columns=drop_cols, inplace=True)

df = df[(df['age'] >= 17) & (df['age'] <= 75)]

df = df[df['workclass'] != 'Without-pay']
df = df[df['workclass'] != 'Never-worked']

for edu in ['1st-4th', '5th-6th', 'Preschool']:
    if 'education' in df.columns:
        df = df[df['education'] != edu]

df.dropna(inplace=True)

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
df.dropna(subset=['income'], inplace=True)

if 'education' in df.columns and 'education-num' in df.columns:
    df.drop(columns=['education'], inplace=True)

X = df.drop('income', axis=1)
y = df['income']

from sklearn.preprocessing import LabelEncoder
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = {}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    results[name] = (pipe, acc)

best_model_name = max(results, key=lambda k: results[k][1])
best_model = results[best_model_name][0]
print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name][1]:.2f})")
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/salary_prediction_model.pkl")