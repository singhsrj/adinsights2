import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AdInsights AI",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .success {
        background-color: #A8E6CF;
        border: 2px solid #69B578;
    }
    .warning {
        background-color: #FFD3B6;
        border: 2px solid #FF8B6A;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("🎯 AdInsights AI")
    st.markdown("### Harness AI to Know Your Audience and Predict Purchases")

# Sidebar with model information
with st.sidebar:
    st.header("🔍 Model Information")
    st.markdown("""
    This application uses a **Decision Tree Classifier** to predict customer purchasing behavior based on:
    - Age
    - Estimated Salary
    
    #### How it works:
    1. Data preprocessing
    2. Feature scaling
    3. Model training
    4. Prediction
    """)
    
    show_details = st.checkbox("Show Technical Details")

# Load and prepare data
@st.cache_data
def load_data():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    return dataset

@st.cache_resource
def prepare_model(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = DecisionTreeClassifier(criterion='log_loss', random_state=0)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    return classifier, sc, accuracy, X_test, y_test

# Load data and prepare model
dataset = load_data()
classifier, sc, accuracy, X_test, y_test = prepare_model(dataset)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Enter Customer Details")
    
    # Input fields with better formatting
    age = st.number_input("Age", 
                         min_value=18, 
                         max_value=100, 
                         value=30,
                         help="Enter customer's age (18-100)")
    
    salary = st.number_input("Estimated Salary ($)", 
                            min_value=10000, 
                            max_value=200000, 
                            value=60000,
                            step=1000,
                            help="Enter customer's estimated annual salary")

    # Prediction button
    if st.button("🔮 Generate Prediction"):
        with st.spinner('Analyzing customer data...'):
            # Add a small delay for better UX
            time.sleep(1)
            
            # Make prediction
            input_data = sc.transform([[age, salary]])
            prediction = classifier.predict(input_data)
            probability = classifier.predict_proba(input_data)
            
            # Display prediction
            st.markdown("### 🎯 Prediction Result")
            if prediction[0] == 1:
                st.markdown(
                    f"""
                    <div class="prediction-box success">
                        <h2>Will Buy! 🎉</h2>
                        <p>Confidence: {probability[0][1]*100:.2f}%</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="prediction-box warning">
                        <h2>Won't Buy 🤔</h2>
                        <p>Confidence: {probability[0][0]*100:.2f}%</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

with col2:
    st.markdown("### 📈 Model Performance")
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    
    if show_details:
        st.markdown("### 🔍 Feature Distribution")
        # Create scatter plot of test data
        df_test = pd.DataFrame(X_test, columns=['Age', 'Salary'])
        df_test['Purchase'] = y_test
        fig = px.scatter(df_test, x='Age', y='Salary', color='Purchase',
                        title='Test Data Distribution',
                        labels={'Purchase': 'Purchase Decision'})
        st.plotly_chart(fig)

# Footer with model details if checkbox is selected
if show_details:
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Preprocessing**
        - Standard Scaling
        - Train-Test Split (75-25)
        """)
    
    with col2:
        st.markdown("""
        **Model Parameters**
        - Criterion: Log Loss
        - Random State: 0
        """)
    
    with col3:
        st.markdown("""
        **Dataset Info**
        - Features: Age, Salary
        - Target: Purchase Decision
        """)