
# 🩺 Diabetes Prediction Web Application

## Empowering Health Insights with Machine Learning & Streamlit

-----

## ✨ Project Overview

This project delivers a **Diabetes Prediction Web Application** built using **Machine Learning** models and deployed with **Streamlit**. Leveraging the well-known PIMA Indian Diabetes Dataset, the application allows users to input various health parameters and receive an immediate prediction of their diabetes risk, along with insightful visualizations of contributing factors.

This solution provides an intuitive interface for healthcare professionals or individuals to quickly assess potential diabetes risk based on diagnostic measurements.

-----

## 🎯 Key Features

  * **Interactive Streamlit Web App**: A user-friendly graphical interface for real-time diabetes risk prediction.
  * **Intuitive Input Sliders**: Easily adjust patient parameters like Glucose, BMI, Age, Blood Pressure, etc.
  * **Dual Model Approach**: Compares Logistic Regression and Random Forest Classifiers to select the best-performing model.
  * **Automated Missing Value Handling**: Imputes missing values using median strategy for robust data preparation.
  * **Comprehensive Evaluation Metrics**: Displays accuracy, confusion matrix, and classification report for model transparency.
  * **Dynamic Feature Importance**: Visualizes the most influential factors driving the prediction for transparency and insight.
  * **Contextual Data Visualization**: Plots patient's glucose and BMI relative to the overall dataset for a clearer understanding.
  * **Model Persistence**: The best-performing model is saved using `joblib` for efficient loading and deployment.

-----

## 📊 Dataset

The project utilizes the **Pima Indians Diabetes Database**, a widely used dataset for binary classification. It contains diagnostic measurements and the outcome (diabetes or not) for female patients of Pima Indian heritage.

**Attributes include:**

  * `Pregnancies`: Number of times pregnant
  * `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
  * `BloodPressure`: Diastolic blood pressure (mm Hg)
  * `SkinThickness`: Triceps skin fold thickness (mm)
  * `Insulin`: 2-Hour serum insulin (mu U/ml)
  * `BMI`: Body mass index (weight in kg/(height in m)^2)
  * `DiabetesPedigreeFunction`: Diabetes pedigree function
  * `Age`: Age in years
  * `Outcome`: Class variable (0: Non-diabetic, 1: Diabetic)

The dataset is loaded directly from a public GitHub raw CSV URL.

-----

## 🚀 Getting Started

Follow these steps to set up the project locally and run the Diabetes Prediction Web Application:

### Prerequisites

  * Python 3.9+
  * Pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/0001Moksh/Diabetes-Prediction-App/tree/main/Diabetes%20Prediction%20App](https://github.com/0001Moksh/Diabetes-Prediction-App/tree/main)
    cd diabetes-prediction-app
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
    ```

### Running the Application

This project involves two main parts: model training and the Streamlit app.

1.  **Train and Save the Model:**
    Before running the web app, you need to train the machine learning model and save it.
    The provided Python code (which you can put into a file like `train_model.py` or run as a Jupyter Notebook) handles:

      * Loading and preprocessing the dataset.
      * Training Logistic Regression and Random Forest models.
      * Evaluating and comparing their performance.
      * Selecting the best model (Logistic Regression in this case) and saving it as `diabetes_model.pkl`.

    Ensure you run this training script/notebook at least once to generate the `diabetes_model.pkl` file.

2.  **Launch the Streamlit Web Application:**
    Once the `diabetes_model.pkl` file is generated, you can launch the interactive app.

    ```bash
    streamlit run app.py
    ```

    Your default web browser will open, displaying the Diabetes Prediction App. You can then adjust the patient parameters using the sliders and click "Predict Diabetes Risk" to see the results.

-----

## 📈 Model Performance

The project evaluates Logistic Regression and Random Forest Classifiers. The best-performing model (Logistic Regression in this specific execution) is selected for deployment.

**Logistic Regression Performance:**

  * **Accuracy:** `~75%`
  * **Classification Report:**
    ```
                  precision    recall  f1-score   support

               0       0.81      0.79      0.80        99
               1       0.64      0.67      0.65        55

        accuracy                           0.75       154
       macro avg       0.73      0.73      0.73       154
    weighted avg       0.75      0.75      0.75       154
    ```

#### Logistic Regression Confusion Matrix

*(Note: You'll need to generate and save this image in your repo, or add a placeholder if you don't save it directly.)*

#### Logistic Regression Feature Importance

*(Note: You'll need to generate and save this image in your repo, or add a placeholder.)*

-----

## 📂 Project Structure

```
.
├── app.py                                   # Streamlit web application code
├── train_model.ipynb                        # Jupyter/Colab notebook for model training (or train_model.py)
├── diabetes_model.pkl                       # Saved best-performing machine learning model
├── README.md                                # This file
```

*(Note: `.pkl` files and images will be generated after running the training code. Remember to add a compelling screenshot of your Streamlit app for the `app_screenshot.png` file\!)*

-----

## 🤝 Contributing

Contributions are highly appreciated\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## 🧑‍💻 Author

**Deva** (AI Assistant created by Moksh Bhardwaj)

  * **Moksh Bhardwaj** - B.Tech AIML Student | DPG ITM College
  * **Location**: Basai Enclave Part 3, Gurugram, Haryana, India

-----
