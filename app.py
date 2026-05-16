import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="E-commerce Customer Analytics",
    page_icon="🛒",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

df = pd.read_csv("final_customer_dashboard_data.csv")

# Original Retail Dataset
retail_df = pd.read_csv("online_retail.csv")

# ---------------- DATA CLEANING ----------------

retail_df = retail_df.dropna(subset=['Customer ID'])
retail_df = retail_df[retail_df['Quantity'] > 0]

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

[data-testid="stSidebar"] {
    background-color: #dbeafe;
}

.title {
    font-size: 60px;
    font-weight: bold;
    text-align: center;
    color: #111827;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3144/3144456.png",
    width=120
)

st.sidebar.title("📌 Filters")

# Segment Filter
segment = st.sidebar.multiselect(
    "Customer Segment",
    options=df['Segment'].unique(),
    default=df['Segment'].unique()
)

# Risk Filter
risk = st.sidebar.multiselect(
    "Risk Level",
    options=df['Risk_Level'].unique(),
    default=df['Risk_Level'].unique()
)

# Status Filter
status = st.sidebar.multiselect(
    "Customer Status",
    options=df['Customer_Status'].unique(),
    default=df['Customer_Status'].unique()
)

# ---------------- FILTER DATA ----------------

filtered_df = df[
    (df['Segment'].isin(segment)) &
    (df['Risk_Level'].isin(risk)) &
    (df['Customer_Status'].isin(status))
]

# ---------------- EMPTY DATA CHECK ----------------

if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# ---------------- HEADER ----------------

st.markdown("""
<h1 style='
text-align: center;
font-size: 60px;
color: #111827;
font-weight: bold;
'>
🛒 E-commerce Customer Behavior Analytics
</h1>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Interactive Customer Analytics Dashboard with AI Recommendation System</p>',
    unsafe_allow_html=True
)

# ---------------- KPI CARDS ----------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Total Customers",
        len(filtered_df)
    )

with col2:
    st.metric(
        "💰 Total Revenue",
        f"₹ {filtered_df['Monetary'].sum():,.0f}"
    )

with col3:

    churn_count = len(
        filtered_df[
            filtered_df['Customer_Status'] == 'Churned'
        ]
    )

    st.metric(
        "⚠️ Churned Customers",
        churn_count
    )

with col4:

    vip_count = len(
        filtered_df[
            filtered_df['Segment'] == 'VIP Customers'
        ]
    )

    st.metric(
        "⭐ VIP Customers",
        vip_count
    )

st.markdown("---")

# ---------------- CHARTS ROW 1 ----------------

col5, col6 = st.columns(2)

# Pie Chart
with col5:

    fig1 = px.pie(
        filtered_df,
        names='Segment',
        title='Customer Segmentation Distribution',
        hole=0.5
    )

    st.plotly_chart(fig1, use_container_width=True)

# Revenue Chart
with col6:

    revenue_df = filtered_df.groupby(
        'Segment'
    )['Monetary'].sum().reset_index()

    fig2 = px.bar(
        revenue_df,
        x='Segment',
        y='Monetary',
        color='Segment',
        title='Revenue by Customer Segment'
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------- CHARTS ROW 2 ----------------

col7, col8 = st.columns(2)

# Scatter Plot
with col7:

    fig3 = px.scatter(
        filtered_df,
        x='Recency',
        y='Monetary',
        color='Risk_Level',
        size='Frequency',
        hover_data=['Segment'],
        title='Customer Risk Analysis'
    )

    st.plotly_chart(fig3, use_container_width=True)

# Risk Level Chart
with col8:

    risk_df = filtered_df.groupby(
        'Risk_Level'
    )['Customer ID'].count().reset_index()

    fig4 = px.bar(
        risk_df,
        x='Risk_Level',
        y='Customer ID',
        color='Risk_Level',
        title='Customer Risk Distribution'
    )

    st.plotly_chart(fig4, use_container_width=True)


# ---------------- AI RECOMMENDATION SYSTEM ----------------

st.markdown("---")

st.subheader("🛍️ AI Product Recommendation System")

# Create Customer Product Matrix
customer_product_matrix = retail_df.pivot_table(
    index='Customer ID',
    columns='Description',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

# Cosine Similarity
similarity = cosine_similarity(customer_product_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=customer_product_matrix.index,
    columns=customer_product_matrix.index
)

# Recommendation Function
def recommend_products(customer_id, top_n=5):

    customer_id = float(customer_id)

    similar_customers = similarity_df.loc[
        customer_id
    ].sort_values(
        ascending=False
    ).iloc[1:6]

    similar_customer_ids = similar_customers.index

    similar_customer_products = customer_product_matrix.loc[
        similar_customer_ids
    ]

    recommended_products = similar_customer_products.sum().sort_values(
        ascending=False
    )

    # Remove already purchased products
    purchased_products = customer_product_matrix.loc[
        customer_id
    ]

    purchased_products = purchased_products[
        purchased_products > 0
    ].index

    recommended_products = recommended_products.drop(
        purchased_products,
        errors='ignore'
    )

    return recommended_products.head(top_n)

# ---------------- CUSTOMER SELECTION ----------------

st.markdown("---")

st.subheader("👤 Customer Analysis & Recommendations")

selected_customer = st.selectbox(
    "Select Customer ID",
    sorted(df['Customer ID'].unique())
)

# Customer Details
customer_data = filtered_df[
    filtered_df['Customer ID'] == selected_customer
]

if not customer_data.empty:

    st.dataframe(customer_data)

else:
    st.warning("Customer data not found")

# Generate Recommendations
try:

    recommendations = recommend_products(
        selected_customer
    )

    recommended_df = pd.DataFrame(
        recommendations
    ).reset_index()

    recommended_df.columns = [
        'Recommended Product',
        'Recommendation Score'
    ]

    st.dataframe(recommended_df)

    # Recommendation Chart
    fig5 = px.bar(
        recommended_df,
        x='Recommendation Score',
        y='Recommended Product',
        orientation='h',
        title='Top Recommended Products'
    )

    st.plotly_chart(fig5, use_container_width=True)

except:
    st.error("Recommendations not available")

# ---------------- HIGH RISK CUSTOMERS ----------------

st.markdown("---")

st.subheader("🚨 High Risk Customers")

high_risk = filtered_df[
    filtered_df['Risk_Level'] == 'High Risk'
]

if high_risk.empty:

    st.info("No High Risk Customers Found")

else:

    st.dataframe(
        high_risk[[
            'Customer ID',
            'Segment',
            'Monetary',
            'Churn_Probability (%)'
        ]].head(10)
    )
# ---------------- POWER BI DASHBOARD ----------------

st.markdown("---")

import streamlit.components.v1 as components

st.subheader("📊 Power BI Dashboard")

power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=c6f71501-5a94-4133-bd0a-20c29b17f337&autoAuth=true&ctid=381d5f7c-efaf-4cd0-a4a4-479a9c0d1792"

components.iframe(
    power_bi_url,
    width=1200,
    height=700,
    scrolling=True
)

# ---------------- DOWNLOAD BUTTON ----------------

st.markdown("---")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Customer Report",
    data=csv,
    file_name='customer_report.csv',
    mime='text/csv'
)

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
    <center>
    <h4>Developed by Shraddha Dilip Mali</h4>
    </center>
    """,
    unsafe_allow_html=True
)