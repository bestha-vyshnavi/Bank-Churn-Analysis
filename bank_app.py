import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏦 Bank Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv('European_Bank (1).csv')
    return df

@st.cache_data
def train_model(df):
    le = LabelEncoder()
    df_model = df.copy()
    df_model['Geography'] = le.fit_transform(df_model['Geography'])
    df_model['Gender'] = le.fit_transform(df_model['Gender'])
    df_model['EngagementScore'] = (df_model['IsActiveMember'] * 2 +
                                    df_model['HasCrCard'] +
                                    df_model['NumOfProducts'])
    features = ['CreditScore','Geography','Gender','Age','Tenure',
                'Balance','NumOfProducts','HasCrCard','IsActiveMember',
                'EstimatedSalary','EngagementScore']
    X = df_model[features]
    y = df_model['Exited']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features, le

df = load_data()
model, acc, features, le = train_model(df)

st.title("🏦 Bank Customer Churn Analytics")
st.subheader("Customer Engagement & Retention Strategy Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
total = len(df)
churned = df['Exited'].sum()
retained = total - churned
high_risk = len(df[(df['Balance'] > df['Balance'].quantile(0.75)) &
                   (df['IsActiveMember'] == 0)])

with col1:
    st.metric("👥 Total Customers", f"{total:,}")
with col2:
    st.metric("⚠️ Churned", f"{churned:,}", f"{churned/total*100:.1f}%")
with col3:
    st.metric("✅ Retained", f"{retained:,}", f"{retained/total*100:.1f}%")
with col4:
    st.metric("🤖 Model Accuracy", f"{acc*100:.1f}%")

st.markdown("---")

st.sidebar.title("🔧 Filters")
geo_filter = st.sidebar.multiselect(
    "Select Geography",
    df['Geography'].unique(),
    default=df['Geography'].unique()
)
gender_filter = st.sidebar.multiselect(
    "Select Gender",
    df['Gender'].unique(),
    default=df['Gender'].unique()
)
active_filter = st.sidebar.radio(
    "Customer Activity",
    ["All", "Active Only", "Inactive Only"]
)

filtered = df[
    (df['Geography'].isin(geo_filter)) &
    (df['Gender'].isin(gender_filter))
]
if active_filter == "Active Only":
    filtered = filtered[filtered['IsActiveMember'] == 1]
elif active_filter == "Inactive Only":
    filtered = filtered[filtered['IsActiveMember'] == 0]

col1, col2 = st.columns(2)
with col1:
    st.subheader("🌍 Churn by Geography")
    geo_churn = filtered.groupby('Geography')['Exited'].mean() * 100
    fig1 = px.bar(x=geo_churn.index, y=geo_churn.values,
                  color=geo_churn.values,
                  color_continuous_scale='RdYlGn_r',
                  labels={'x': 'Country', 'y': 'Churn Rate %'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("📦 Churn by Products")
    prod_churn = filtered.groupby('NumOfProducts')['Exited'].mean() * 100
    fig2 = px.bar(x=prod_churn.index, y=prod_churn.values,
                  color=prod_churn.values,
                  color_continuous_scale='RdYlGn_r',
                  labels={'x': 'Number of Products', 'y': 'Churn Rate %'})
    st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Age vs Churn")
    fig3 = px.histogram(filtered, x='Age', color='Exited',
                        barmode='overlay',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("💰 Balance vs Churn")
    fig4 = px.box(filtered, x='Exited', y='Balance',
                  color='Exited',
                  color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("🔥 Engagement Heatmap")
pivot = filtered.pivot_table(
    values='Exited',
    index='Geography',
    columns='IsActiveMember',
    aggfunc='mean'
) * 100
pivot.columns = ['Inactive', 'Active']
fig5 = px.imshow(pivot, color_continuous_scale='RdYlGn_r',
                 text_auto='.1f', aspect='auto')
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.subheader("🤖 Churn Risk Predictor")

col1, col2, col3 = st.columns(3)
with col1:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance ($)", 0, 300000, 50000)

with col2:
    geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    salary = st.number_input("Estimated Salary ($)", 0, 200000, 50000)

with col3:
    has_cc = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])
    predict_btn = st.button("🔍 Predict Churn Risk", type="primary")

if predict_btn:
    geo_enc = {'France': 0, 'Germany': 1, 'Spain': 2}
    gen_enc = {'Female': 0, 'Male': 1}
    cc = 1 if has_cc == "Yes" else 0
    active = 1 if is_active == "Yes" else 0
    eng_score = (active * 2) + cc + num_products

    input_data = [[credit_score, geo_enc[geography], gen_enc[gender],
                   age, tenure, balance, num_products, cc, active,
                   salary, eng_score]]

    prob = model.predict_proba(input_data)[0][1]
    risk = "🔴 HIGH RISK" if prob > 0.5 else "🟢 LOW RISK"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.metric("Risk Level", risk)
    with col2:
        fig6 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Churn Risk %"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if prob > 0.5 else "green"},
                   'steps': [
                       {'range': [0, 30], 'color': '#2ecc71'},
                       {'range': [30, 60], 'color': '#f39c12'},
                       {'range': [60, 100], 'color': '#e74c3c'}
                   ]}))
        st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")
st.success("✅ Dashboard by Vyshnavi Bestha | Bank Churn Analytics | 2026")