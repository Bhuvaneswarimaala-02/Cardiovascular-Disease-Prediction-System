# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import streamlit as st

# # %% Load the dataset
# df = pd.read_csv('Dataset.csv')

# # %% Data Preparation (Select only 6 features for prediction)
# selected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
# X = df[selected_features]
# y = df['target']

# # %% Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # %% Feature Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # %% Model Function (Decision Tree only)
# def train_model():
#     # Initialize the Decision Tree model
#     model = DecisionTreeClassifier()
#     model.fit(X_train, y_train)
#     return model

# # %% Streamlit UI
# def main():
#     st.title("Cardiovascular Disease Prediction")
#     st.markdown("Enter the details below to check if you are at risk.")
    
#     # Input fields
#     age = st.number_input("Age", min_value=18, max_value=100, value=50)
#     sex = st.radio("Sex", ("Male", "Female"))
#     # cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
#     # Mapping of chest pain types
#     cp_options = {
#         "Typical Angina": 0,
#         "Atypical Angina": 1,
#         "Non-anginal Pain": 2,
#         "Asymptomatic": 3
#     }

#     cp_label = st.selectbox("Chest Pain Type", list(cp_options.keys()))
#     cp = cp_options[cp_label]  # Get the corresponding numeric value

#     trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
#     chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
#     thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    
#     # Convert categorical inputs to numerical values
#     sex = 1 if sex == "Male" else 0
    
#     # Load the model only when the "Predict" button is clicked
#     if st.button("Predict"): 
#         # Train the model again if it's not trained yet (you can also load a pre-trained model)
#         model = train_model()

#         # Get the user input and predict
#         input_data = np.array([[age, sex, cp, trestbps, chol, thalach]])
#         input_data = scaler.transform(input_data)  # Apply scaling to input data
#         prediction = model.predict(input_data)[0]
        
#         if prediction == 1:
#             st.error("High risk! You may have cardiovascular disease.")
#         else:
#             st.success("Low risk. You are unlikely to have cardiovascular disease.")

# # %% Run the app
# if __name__ == "__main__":
#     main()






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Load dataset
df = pd.read_csv('Dataset.csv')

# Target and features
target_column = 'target'
feature_columns = [col for col in df.columns if col != target_column]
X = df[feature_columns]
y = df[target_column]

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
def train_model():
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Field label descriptions
field_labels = {
    'age': 'Age (in years)',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure (mm Hg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120 mg/dl',
    'restecg': 'Resting Electrocardiogram Results',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression Induced by Exercise',
    'slope': 'Slope of the Peak Exercise ST Segment',
    'ca': 'Number of Major Vessels (0-3)',
    'thal': 'Thalassemia'
}

# Categorical options with human-readable labels
sex_options = {"Male": 1, "Female": 0}

cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

fbs_options = {
    "Fasting Blood Sugar > 120 mg/dl (True)": 1,
    "Fasting Blood Sugar ≤ 120 mg/dl (False)": 0
}

restecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

exang_options = {
    "Yes": 1,
    "No": 0
}

slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

thal_options = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

# Streamlit App
def main():
    st.title("Cardiovascular Disease Prediction")
    st.markdown("Please enter the following details:")

    user_input = []

    for col in feature_columns:
        label = field_labels.get(col, col)

        if col == 'sex':
            selected = st.selectbox(label, list(sex_options.keys()))
            user_input.append(sex_options[selected])

        elif col == 'cp':
            selected = st.selectbox(label, list(cp_options.keys()))
            user_input.append(cp_options[selected])

        elif col == 'fbs':
            selected = st.selectbox(label, list(fbs_options.keys()))
            user_input.append(fbs_options[selected])

        elif col == 'restecg':
            selected = st.selectbox(label, list(restecg_options.keys()))
            user_input.append(restecg_options[selected])

        elif col == 'exang':
            selected = st.selectbox(label, list(exang_options.keys()))
            user_input.append(exang_options[selected])

        elif col == 'slope':
            selected = st.selectbox(label, list(slope_options.keys()))
            user_input.append(slope_options[selected])

        elif col == 'thal':
            selected = st.selectbox(label, list(thal_options.keys()))
            user_input.append(thal_options[selected])

        elif col == 'ca':
            val = st.selectbox(label, sorted(df['ca'].dropna().unique().astype(int)))
            user_input.append(val)

        else:
            val = st.number_input(label, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            user_input.append(val)

    input_df = pd.DataFrame([user_input], columns=feature_columns)
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        model = train_model()
        prediction = model.predict(input_scaled)[0]
        if prediction == 1:
            st.error("High risk! You may have cardiovascular disease.")
        else:
            st.success("Low risk. You are unlikely to have cardiovascular disease.")

if __name__ == "__main__":
    main()
