
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title('🔮 Diabetes Prediction App')
st.sidebar.header('📋 Patient Data')


data_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.subheader('📊 Dataset Overview')
    st.write(df.head())

    if 'Outcome' not in df.columns:
        st.error("❌ Error: 'Outcome' column not found in uploaded dataset.")
    else:
        st.success("✅ 'Outcome' column found! Proceeding...")
        
        df = df.dropna()
    
        st.subheader('📈 Data Statistics')
        st.write(df.describe())

        X = df.drop(columns=['Outcome'])
        y = df['Outcome']

     
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

       
        rf_model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')
        rf_model.fit(X_train, y_train)

      
        def user_report():
            user_data = {}
            for col in X.columns:
                min_val, max_val = df[col].min(), df[col].max()
                user_data[col] = st.sidebar.slider(f"{col}", float(min_val), float(max_val), float(df[col].median()))
            return pd.DataFrame(user_data, index=[0])

        user_data = user_report()

        if user_data is not None:
            st.subheader('📋 Patient Data Entered')
            st.write(user_data)

           
            user_data_scaled = scaler.transform(user_data)

            user_result = rf_model.predict(user_data_scaled)

            # Prediction Result
            st.subheader('🔮 Diabetes Prediction Result:')
            if user_result[0] == 1:
                st.error('⚠️ You are likely Diabetic')
            else:
                st.success('✅ You are likely Not Diabetic')

       
            accuracy = accuracy_score(y_test, rf_model.predict(X_test))
            st.subheader('📈 Model Accuracy:')
            st.write(f"🎯 {accuracy * 100:.2f}%")

         
            st.subheader('🧠 Comparing Your Data with Dataset')
            for feature in X.columns:
                fig, ax = plt.subplots()
                sns.histplot(df[feature], kde=True, color='gray', label='Dataset', ax=ax)
                ax.axvline(user_data[feature].values[0], color='red', linestyle='dashed', linewidth=2, label='Your Value')
                ax.legend()
                st.pyplot(fig)



