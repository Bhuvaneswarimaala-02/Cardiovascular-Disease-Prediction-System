import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# %% Load the dataset
df = pd.read_csv('Dataset.csv')

# %% Data Preparation (Select only 6 features for prediction)
selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
X = df[selected_features]
y = df['target']

# %% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Model Function (Decision Tree only)
def train_model():
    # Initialize the Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# %% Streamlit UI
def main():
    st.title("Cardiovascular Disease Prediction")
    st.markdown("Enter the details below to check if you are at risk.")
    
    # Input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    sex = st.radio("Sex", ("Male", "Female"))
    # cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    # Mapping of chest pain types
    cp_options = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }

    cp_label = st.selectbox("Chest Pain Type", list(cp_options.keys()))
    cp = cp_options[cp_label]  # Get the corresponding numeric value

    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
    # Convert categorical inputs to numerical values
    sex = 1 if sex == "Male" else 0
    
    # Load the model only when the "Predict" button is clicked
    if st.button("Predict"): 
        # Train the model again if it's not trained yet (you can also load a pre-trained model)
        model = train_model()

        # Get the user input and predict
        input_data = np.array([[age, sex, cp, trestbps, chol, thalach]])
        input_data = scaler.transform(input_data)  # Apply scaling to input data
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            st.error("High risk! You may have cardiovascular disease.")
        else:
            st.success("Low risk. You are unlikely to have cardiovascular disease.")

# %% Run the app
if __name__ == "__main__":
    main()
