# Salary Prediction & Insights Web App

A Streamlit-powered web application for predicting salary categories and exploring insights from the UCI Adult Income dataset. Includes a simple rule-based chatbot for interactive data queries.

---

## Features

- **Salary Prediction:**  
  Enter demographic and work details to predict if a person's salary is likely to be >$50K or ≤$50K using a trained machine learning model.

- **Salary Insights Dashboard:**  
  Visualize salary distributions by workclass, occupation, education, and more. Explore average hours worked and income group proportions.

- **Interactive Chatbot:**  
  Ask questions about the dataset, salary trends, occupations, education, and app usage. The chatbot provides instant, data-driven answers.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Intership
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your default web browser.

---

## Project Structure

```
Intership/
│
├── app.py                # Streamlit web app
├── train_model.py        # Model training and saving script
├── models/
│   └── salary_prediction_model.pkl
├── data/
│   └── adult 3.csv       # Cleaned dataset
├── requirements.txt      # Python dependencies
└── README.md
```

---

## How It Works

- **Data Cleaning:**  
  Handles missing values, removes outliers, and encodes categorical features for robust model training.

- **Model Training:**  
  Compares multiple classifiers (Random Forest, Logistic Regression, Decision Tree, Gradient Boosting, KNN) and saves the best model.

- **Prediction:**  
  User inputs are encoded and passed to the trained model for salary prediction.

- **Insights:**  
  Uses Seaborn and Matplotlib for interactive charts and summaries.

- **Chatbot:**  
  Answers queries like:
  - Which occupation pays the most?
  - What is the average salary(prediction) of Exec-managerial?
  - Does education affect salary?
  - How many people earn >50K?
  - List all occupations or education levels.

---

## Example Usage

- **Predict Salary:**  
  Fill in the sidebar form and click "Predict" to see your result.

- **Explore Insights:**  
  Switch to "Salary Insights" for charts and statistics.

- **Ask the Chatbot:**  
  Go to "Chatbot" and type questions such as:
  - `What is the average salary of Machine-op-inspct?`
  - `Does education affect salary?`
  - `List occupations`

---

## Author

Developed by P.Namish as a beginner-friendly machine learning and data visualization project.

---

## License

This project is for educational purposes.
