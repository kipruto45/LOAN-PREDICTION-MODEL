
🏦 Loan Prediction Model

This project implements a Loan Prediction Model that uses machine learning techniques to predict the likelihood of a loan being approved based on applicant details. It aims to assist banks and financial institutions in automating the loan eligibility process.

📌 Features

Predicts loan approval (Yes/No) based on user input

Trained using historical loan application data

Preprocessing pipeline for handling missing values and categorical variables

Supports logistic regression, decision tree, and random forest models

Web interface for user input (optional)

📁 Project Structure

loan-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── loan_model.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction_service.py
├── app/
│   └── app.py  # Flask or Streamlit app
├── README.md
└── requirements.txt


---

🔧 Installation

1. Clone the repo:

git clonehttps://github.com/kipruto45/LOAN-PREDICTION-MODEL/git.
cd loan-prediction


2. Create a virtual environment (optional):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


3. Install dependencies:

pip install -r requirements.txt

🚀 Usage

1. Train the Model

python src/model_training.py

This will train the model and save it to the models/ directory.

2. Run the Web App (Optional)

If you have a frontend app using Streamlit or Flask:

streamlit run app/app.py

or

python app/app.py

3. Make Predictions (Programmatic Use)

from src.prediction_service import predict_loan_status

sample_input = {
    'Gender': 'Male',
    'Married': 'Yes',
    'Education': 'Graduate',
    'ApplicantIncome': 5000,
    'LoanAmount': 200,
    ...
}

result = predict_loan_status(sample_input)
print("Prediction:", result)

📊 Dataset

Source: Loan Prediction Dataset - Analytics Vidhya

Features include:

Gender, Married, Education, Self_Employed

ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

Property_Area

Target: Loan_Status (Y/N)
🧠 Model Performance

Accuracy: ~80% on validation set

Cross-validation used for model selection and tuning

Evaluation metrics: Accuracy, Precision, Recall, F1-score


🛠 Tech Stack

Python 3.x

Pandas, NumPy

Scikit-learn

Streamlit / Flask (for app)

Joblib / Pickle for model serialization

📝 License

This project is open-source under the MIT License.
🙋‍♂️ Contact

For questions or suggestions, please open an issue or contact ropellroe@gmail.com

