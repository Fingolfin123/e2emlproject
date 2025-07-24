# 📊 Student Score Prediction App

This is a Flask-based web application that allows users to:

- **Train** a machine learning model based on selected features from student performance data.
- **Predict** a student's math score using user inputs for categorical and numerical features.

---

## 🚀 Features

### 1. Train a Model
- Select a single feature (e.g., gender, lunch type, reading score) from a dropdown.
- The app trains and evaluates several ML models (e.g., Linear Regression, Random Forest, etc.).
- The top-performing model is saved automatically for use in predictions.

### 2. Make a Prediction
- Input student details like gender, parental education, and test scores.
- The app returns a predicted **math score** using the best-trained model.

---

## 🏗️ Project Structure



# End to End Machine Learning Project

<!-- 1. Create Env -->
>> conda create -p venv python==3.8 -y
>> conda activate venv/


<!-- 2. Init Git Repo -->
>> git init
>> create a README (here!)

<!-- 3. Commit README.md to local -->
>> git add README.md
>> git commit -m "First Commit"

<!-- 4. Push ReadM.md to Source -->
>> git branch -M main
>> git remote add origin https://github.com/Fingolfin123/e2emlproject.git

>> git push -u origin main

<!-- 5. Create src folder and pyproject.toml -->

<!-- 6. Create conda venv -->
conda create -n yourenv pip
conda activate yourenv/

<!-- 7. Install env package --> 
pip install -e . --timeout 100 <!-- extend timeout to avoid network issues>
