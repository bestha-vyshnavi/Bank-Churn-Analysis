import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏦 Bank Churn Analytics", page_icon="🏦", layout="wide")

@st.cache_data
def load_data():
    if True:
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'CreditScore': np.random.randint(300, 850, n),
            'Geography': np.random.choice(['France', 'Germany', 'Spain'], n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'Age': np.random.randint(18, 92, n),
            'Tenure': np.random.randint(0, 10, n),
            'Balance': np.random.randint(0, 250000, n),
            'NumOfProducts': np.random.choice([1,2,3,4], n, p=[0.5,0.45,0.03,0.02]),
            'HasCrCard': np.random.choice([0,1], n),
            'IsActiveMember': np.random.choice([0,1], n),
            'EstimatedSalary': np.random.randint(10000, 200000, n),
            'Exited': np.random.choice([0,1], n, p=[0.8,0.2])
        })
        return df

@st.cache_data
def train_model(df):
    le = LabelEncoder()
    df_model = df.copy()
    df_model['Geography'] = le.fit_transform(df_model['Geography'])
    df_model['Gender'] = le.fit_transform(df_model['Gender'])
    df_model['EngagementScore'] = (df_model['IsActiveMember']*2 + df_model['HasCrCard'] + df_model['NumOfProducts'])
    features = ['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','EngagementScore']
    X = df_model[features]
    y = df_model['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features

df = load_data()
model, acc, features = train_model(df)

st.title("🏦 Bank Customer Churn Analytics")
st.subheader("Customer Engagement & Retention Strategy Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
total = len(df)
churned = df['Exited'].sum()
with col1:
    st.metric("👥 Total Customers", f"{total:,}")
with col2:
    st.metric("⚠️ Churned", f"{churned:,}", f"{churned/total*100:.1f}%")
with col3:
    st.metric("✅ Retained", f"{total-churned:,}")
with col4:
    st.metric("🤖 Model Accuracy", f"{acc*100:.1f}%")

st.markdown("---")
st.sidebar.title("🔧 Filters")
geo_filter = st.sidebar.multiselect("Geography", df['Geography'].unique(), default=list(df['Geography'].unique()))
active_filter = st.sidebar.radio("Activity", ["All", "Active Only", "Inactive Only"])

filtered = df[df['Geography'].isin(geo_filter)]
if active_filter == "Active Only":
    filtered = filtered[filtered['IsActiveMember']==1]
elif active_filter == "Inactive Only":
    filtered = filtered[filtered['IsActiveMember']==0]

col1, col2 = st.columns(2)
with col1:
    st.subheader("🌍 Churn by Geography")
    geo_churn = filtered.groupby('Geography')['Exited'].mean()*100
    st.bar_chart(geo_churn)

with col2:
    st.subheader("📦 Churn by Products")
    prod_churn = filtered.groupby('NumOfProducts')['Exited'].mean()*100
    st.bar_chart(prod_churn)

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Age Distribution")
    age_data = pd.DataFrame({
        'Churned': filtered[filtered['Exited']==1]['Age'].value_counts().sort_index(),
        'Retained': filtered[filtered['Exited']==0]['Age'].value_counts().sort_index()
    })
    st.line_chart(age_data)

with col2:
    st.subheader("🔥 Activity vs Churn")
    act_churn = filtered.groupby('IsActiveMember')['Exited'].mean()*100
    act_churn.index = ['Inactive', 'Active']
    st.bar_chart(act_churn)

st.markdown("---")
st.subheader("🤖 Churn Risk Predictor")

col1, col2, col3 = st.columns(3)
with col1:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance ($)", 0, 300000, 50000)
with col2:
    geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    num_products = st.selectbox("Products", [1,2,3,4])
    salary = st.number_input("Salary ($)", 0, 200000, 50000)
with col3:
    has_cc = st.radio("Credit Card?", ["Yes", "No"])
    is_active = st.radio("Active Member?", ["Yes", "No"])
    predict_btn = st.button("🔍 Predict Churn Risk", type="primary")

if predict_btn:
    geo_enc = {'France':0, 'Germany':1, 'Spain':2}
    gen_enc = {'Female':0, 'Male':1}
    cc = 1 if has_cc=="Yes" else 0
    active = 1 if is_active=="Yes" else 0
    eng = (active*2) + cc + num_products
    input_data = [[credit_score, geo_enc[geography], gen_enc[gender], age, tenure, balance, num_products, cc, active, salary, eng]]
    prob = model.predict_proba(input_data)[0][1]
    risk = "🔴 HIGH RISK" if prob > 0.5 else "🟢 LOW RISK"
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.metric("Risk Level", risk)
    with col2:
        st.progress(prob)
        if prob > 0.5:
            st.error(f"⚠️ This customer has {prob*100:.1f}% chance of churning!")
        else:
            st.success(f"✅ This customer is likely to stay! ({prob*100:.1f}% churn risk)")

st.markdown("---")
st.success("✅ Dashboard by Vyshnavi Bestha | Bank Churn Analytics | 2026")
