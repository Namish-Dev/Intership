import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

model = joblib.load("models/salary_prediction_model.pkl")

df = pd.read_csv("data/adult 3.csv")
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

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Salary", "Salary Insights", "Chatbot"])

if page == "Predict Salary":
    st.title("Salary Prediction Explorer")
    age = st.slider("Age", int(df["age"].min()), int(df["age"].max()), int(df["age"].median()))
    workclass = st.selectbox("Workclass", sorted(df["workclass"].unique()))
    occupation = st.selectbox("Occupation", sorted(df["occupation"].unique()))
    hours_per_week = st.slider("Hours per Week", int(df["hours-per-week"].min()), int(df["hours-per-week"].max()), int(df["hours-per-week"].median()))
    native_country = st.selectbox("Country", sorted(df["native-country"].unique()))
    if "education-num" in df.columns:
        education_num = st.selectbox("Education-num", sorted(df["education-num"].unique()))
        input_data = pd.DataFrame({
            "age": [age],
            "workclass": [workclass],
            "education-num": [education_num],
            "occupation": [occupation],
            "hours-per-week": [hours_per_week],
            "native-country": [native_country]
        })
    else:
        education = st.selectbox("Education", sorted(df["education"].unique()))
        input_data = pd.DataFrame({
            "age": [age],
            "workclass": [workclass],
            "education": [education],
            "occupation": [occupation],
            "hours-per-week": [hours_per_week],
            "native-country": [native_country]
        })
    from sklearn.preprocessing import LabelEncoder
    for col in input_data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(df[col])
        input_data[col] = le.transform(input_data[col])
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.success(f"Predicted Salary: > $50K (Probability: {prob:.2f})")
        else:
            st.warning(f"Predicted Salary: ≤ $50K (Probability: {1 - prob:.2f})")

elif page == "Salary Insights":
    st.title("Salary Insights Dashboard")
    df['income_label'] = df['income'].map({0: '<=50K', 1: '>50K'})
    st.subheader("Salary Distribution by Workclass")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="workclass", hue="income_label", ax=ax)
    ax.set_title("Workclass vs. Income")
    ax.set_xlabel("Workclass")
    ax.set_ylabel("Count")
    ax.legend(title="Income")
    st.pyplot(fig)
    st.subheader("Top 10 Occupations by Income")
    fig, ax = plt.subplots(figsize=(12, 6))
    top_occupations = df["occupation"].value_counts().index[:10]
    sns.countplot(data=df[df["occupation"].isin(top_occupations)], x="occupation", hue="income_label", order=top_occupations, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Top Occupations vs. Income")
    ax.set_xlabel("Occupation")
    ax.set_ylabel("Count")
    ax.legend(title="Income")
    st.pyplot(fig)
    st.subheader("Salary Distribution by Education Level")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="education", hue="income_label", order=df["education"].value_counts().index, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Education vs. Income")
    ax.set_xlabel("Education")
    ax.set_ylabel("Count")
    ax.legend(title="Income")
    st.pyplot(fig)
    st.subheader("Average Hours Worked per Week by Income")
    avg_hours = df.groupby("income_label")["hours-per-week"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=avg_hours, x="income_label", y="hours-per-week", palette="viridis", ax=ax)
    for i, row in avg_hours.iterrows():
        ax.text(i, row["hours-per-week"] + 0.5, f"{row['hours-per-week']:.1f}", ha='center')
    ax.set_xlabel("Income")
    ax.set_ylabel("Average Hours per Week")
    ax.set_title("Average Hours Worked per Week by Income Group")
    st.pyplot(fig)
    st.subheader("Income Group Proportion in Dataset")
    income_counts = df['income_label'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff9999"])
    ax.set_title("Proportion of Income Groups")
    st.pyplot(fig)

elif page == "Chatbot":
    st.title("Simple Rule-Based Chatbot")
    st.write("Ask a question about the salary prediction app or dataset!")
    st.markdown("""
    **Sample questions you can ask:**
    - Which occupation pays the most?
    - What is the average salary of a particular occupation? (e.g., average salary for Exec-managerial occupation)
    - Does education affect salary?
    - How many people earn >50K?
    - How many people earn <=50K?
    - What are the average hours worked per week for >50K or <=50K?
    - What features/columns are used?
    - What algorithms/models are used?
    - How to use the app?
    """)
    import difflib
    user_input = st.text_input("You:")
    response = ""
    if user_input:
        user_input_lower = user_input.lower().strip()
        def normalize(s):
            return s.lower().replace(" ", "").replace("-", "")
        if "average salary" in user_input_lower or ("probability" in user_input_lower and "occupation" in user_input_lower) or ("what is the average salary of" in user_input_lower):
            found = False
            cleaned_input = user_input_lower.replace("average salary of","").replace("average salary for","").replace("probability of","").replace("probability for","").replace("occupation","").replace("what is the average salary of","").strip()
            occ_list = [str(occ) for occ in df["occupation"].unique()]
            cleaned_norm = normalize(cleaned_input)
            for occ in occ_list:
                if normalize(occ) == cleaned_norm:
                    avg = df[df["occupation"] == occ]["income"].mean()
                    response = f"The average probability of >50K for '{occ}' is {avg*100:.1f}%."
                    found = True
                    break
            if not found:
                matches = difflib.get_close_matches(cleaned_norm, [normalize(o) for o in occ_list], n=1, cutoff=0.6)
                if matches:
                    matched_occ = next((o for o in occ_list if normalize(o) == matches[0]), None)
                    if matched_occ:
                        avg = df[df["occupation"] == matched_occ]["income"].mean()
                        response = f"The average probability of >50K for '{matched_occ}' is {avg*100:.1f}%."
                        found = True
            if not found:
                for occ in occ_list:
                    if cleaned_norm in normalize(occ):
                        avg = df[df["occupation"] == occ]["income"].mean()
                        response = f"The average probability of >50K for '{occ}' is {avg*100:.1f}%."
                        found = True
                        break
            if not found:
                response = "Please specify a valid occupation from the dataset. For example: 'average salary of Exec-managerial'."
        elif any(greet in user_input_lower for greet in ["hello", "hi", "hey"]):
            response = "Hello! How can I help you with the salary prediction app or dataset? You can ask about occupations, education, salary stats, or app usage."
        elif "average salary" in user_input_lower and "education" in user_input_lower:
            found = False
            cleaned_input = user_input_lower.replace("average salary of","").replace("average salary for","").replace("education","").strip()
            edu_list = [str(edu) for edu in df["education"].unique()] if "education" in df.columns else []
            cleaned_norm = normalize(cleaned_input)
            for edu in edu_list:
                if normalize(edu) == cleaned_norm:
                    avg = df[df["education"] == edu]["income"].mean()
                    response = f"The average probability of >50K for '{edu}' education is {avg*100:.1f}%."
                    found = True
                    break
            if not found and edu_list:
                matches = difflib.get_close_matches(cleaned_norm, [normalize(e) for e in edu_list], n=1, cutoff=0.6)
                if matches:
                    matched_edu = next((e for e in edu_list if normalize(e) == matches[0]), None)
                    if matched_edu:
                        avg = df[df["education"] == matched_edu]["income"].mean()
                        response = f"The average probability of >50K for '{matched_edu}' education is {avg*100:.1f}%."
                        found = True
            if not found:
                response = "Please specify a valid education level from the dataset. For example: 'average salary for Bachelors education'."
        elif ("education" in user_input_lower and "effect" in user_input_lower) or ("education" in user_input_lower and "salary" in user_input_lower):
            if "education" in df.columns:
                edu_income = df.groupby("education")["income"].mean().sort_values(ascending=False)
                top_edu = edu_income.index[0]
                top_val = edu_income.iloc[0]
                response = f"Yes, education affects salary. For example, '{top_edu}' has the highest proportion of >50K earners ({top_val*100:.1f}%)."
            else:
                response = "Education column is not available in the current dataset."
        elif (">50k" in user_input_lower or "more than 50k" in user_input_lower) and ("how many" in user_input_lower or "count" in user_input_lower):
            count = (df["income"]==1).sum()
            response = f"There are {count} people in the dataset earning >50K."
        elif ("<=50k" in user_input_lower or "less than 50k" in user_input_lower) and ("how many" in user_input_lower or "count" in user_input_lower):
            count = (df["income"]==0).sum()
            response = f"There are {count} people in the dataset earning <=50K."
        elif (">50k" in user_input_lower or "more than 50k" in user_input_lower) and ("average hours" in user_input_lower or "work" in user_input_lower):
            avg = df[df["income"]==1]["hours-per-week"].mean()
            response = f"People earning >50K work on average {avg:.1f} hours per week."
        elif ("<=50k" in user_input_lower or "less than 50k" in user_input_lower) and ("average hours" in user_input_lower or "work" in user_input_lower):
            avg = df[df["income"]==0]["hours-per-week"].mean()
            response = f"People earning <=50K work on average {avg:.1f} hours per week."
        elif "list occupations" in user_input_lower or "show occupations" in user_input_lower:
            occs = ', '.join(sorted(df["occupation"].unique()))
            response = f"Occupations in the dataset: {occs}"
        elif "list education" in user_input_lower or "show education" in user_input_lower:
            if "education" in df.columns:
                edus = ', '.join(sorted(df["education"].unique()))
                response = f"Education levels in the dataset: {edus}"
            else:
                response = "Education column is not available in the current dataset."
        elif "columns" in user_input_lower or "features" in user_input_lower or "input" in user_input_lower:
            response = f"The model uses: {', '.join(df.columns.drop('income'))}."
        elif "algorithm" in user_input_lower or "model" in user_input_lower:
            response = "The app compares Random Forest, Logistic Regression, Decision Tree, Gradient Boosting, and KNN."
        elif "visualization" in user_input_lower or "chart" in user_input_lower:
            response = "Go to the 'Salary Insights' page to see charts about the data and predictions."
        elif "how to use" in user_input_lower or "predict" in user_input_lower:
            response = "Go to the 'Predict Salary' page, fill in the details, and click Predict to see the result."
        elif "thank" in user_input_lower:
            response = "You're welcome!"
        elif "author" in user_input_lower or "creator" in user_input_lower:
            response = "This app was created as a beginner project for salary prediction. BY P.Namish"
        elif "help" in user_input_lower:
            response = "You can ask about occupations, education, salary stats, features, or app usage. Try: 'Which occupation pays the most?', 'Average salary of Exec-managerial', 'List occupations', 'List education', etc."
        else:
            response = "Sorry, I can only answer questions about the app and dataset. Try asking about occupations, education, salary stats, or app usage."
    if response:
        st.info(response)

