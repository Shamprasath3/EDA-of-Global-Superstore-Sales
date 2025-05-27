import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit page config (must be the very first Streamlit command)
st.set_page_config(page_title="Global Superstore EDA Dashboard", layout="wide")

# Title and description
st.title("📊 Global Superstore EDA Dashboard")
st.markdown("Explore sales, profit, discounts, and trends interactively!")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Sham prasath K\LEARNING\GUVI\EDA_Global Superstore\Sample - Superstore.csv", encoding='ISO-8859-1')
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Customer Name'], inplace=True)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
region_filter = st.sidebar.multiselect("Select Region(s)", options=df['Region'].unique(), default=df['Region'].unique())
category_filter = st.sidebar.multiselect("Select Category(s)", options=df['Category'].unique(), default=df['Category'].unique())

filtered_df = df[(df['Region'].isin(region_filter)) & (df['Category'].isin(category_filter))]

# Display raw data if needed
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_df)

numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']

# Section 1: Univariate Exploration
st.header("1. Univariate Exploration")

# Region order distribution - Plotly strip plot
fig_region = px.strip(filtered_df, x='Region', title='📌 Order Distribution by Region')
st.plotly_chart(fig_region, use_container_width=True)

# Category order distribution - violin plot
fig_category = px.violin(filtered_df, y='Category', box=True, points="all", title='🛒 Distribution of Orders across Categories')
st.plotly_chart(fig_category, use_container_width=True)

# Sales distribution - histogram + boxplot
fig_sales = px.histogram(filtered_df, x='Sales', nbins=30, marginal='box', title='💰 Sales Distribution with Boxplot')
st.plotly_chart(fig_sales, use_container_width=True)

# Profit distribution - violin
fig_profit = px.violin(filtered_df, y='Profit', box=True, points="all", title='📈 Profit Distribution (Violin Plot)')
st.plotly_chart(fig_profit, use_container_width=True)


# Section 2: Bivariate & Multivariate Exploration
st.header("2. Bivariate & Multivariate Exploration")

# Sales by Category - box plot
fig_sales_cat = px.box(filtered_df, x='Category', y='Sales', title='📦 Sales Distribution by Category')
st.plotly_chart(fig_sales_cat, use_container_width=True)

# Profit vs Discount scatter with trendline
fig_profit_disc = px.scatter(filtered_df, x='Discount', y='Profit', color='Category',
                             title='💸 Profit vs Discount by Category', trendline='ols')
st.plotly_chart(fig_profit_disc, use_container_width=True)

# Correlation heatmap (matplotlib + seaborn)
st.subheader("🔗 Correlation Matrix")
fig_corr, ax = plt.subplots(figsize=(6,4))
corr = filtered_df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

# Profit by Region and Category - stacked bar
st.subheader("📊 Profit by Region and Category")
pivot = filtered_df.pivot_table(values='Profit', index='Region', columns='Category', aggfunc='sum')
fig_profit_region = go.Figure()
for col in pivot.columns:
    fig_profit_region.add_trace(go.Bar(name=col, x=pivot.index, y=pivot[col]))
fig_profit_region.update_layout(barmode='stack')
st.plotly_chart(fig_profit_region, use_container_width=True)

# Section 3: Time Series Analysis
st.header("3. Time Series Analysis")

monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly_sales['Order Month'] = monthly_sales['Order Date'].dt.to_timestamp()

# Monthly Sales Trend
fig_monthly = px.line(monthly_sales, x='Order Month', y='Sales', markers=True, title='📅 Monthly Sales Trend')
st.plotly_chart(fig_monthly, use_container_width=True)

# Seasonal Sales Variation (area chart)
fig_seasonal = px.area(monthly_sales, x='Order Month', y='Sales', title='📆 Seasonal Sales Variation')
st.plotly_chart(fig_seasonal, use_container_width=True)

# Section 4: Summary of Key Insights
st.header("4. Summary of Key Insights")

# Most profitable category
most_prof_cat = filtered_df.groupby('Category')['Profit'].sum().reset_index()
fig_most_prof_cat = px.bar(most_prof_cat, x='Category', y='Profit', color='Profit', title='🏆 Most Profitable Category')
st.plotly_chart(fig_most_prof_cat, use_container_width=True)
st.markdown(f"**Most profitable category:** {most_prof_cat.loc[most_prof_cat['Profit'].idxmax(), 'Category']} 👏")

# Least profitable region
least_prof_region = filtered_df.groupby('Region')['Profit'].sum().reset_index()
fig_least_prof_reg = px.bar(least_prof_region, x='Region', y='Profit', color='Profit', title='📉 Least Profitable Region')
st.plotly_chart(fig_least_prof_reg, use_container_width=True)
st.markdown(f"**Least profitable region:** {least_prof_region.loc[least_prof_region['Profit'].idxmin(), 'Region']} 😟")

# Discount bins vs average profit
filtered_df['Discount Bin'] = pd.cut(filtered_df['Discount'], bins=[0,0.1,0.2,0.3,0.4,0.5,1.0])
filtered_df['Discount Bin'] = filtered_df['Discount Bin'].astype(str)
discount_profit = filtered_df.groupby('Discount Bin')['Profit'].mean().reset_index()
fig_discount_profit = px.bar(discount_profit, x='Discount Bin', y='Profit', title='⚠️ Average Profit by Discount Range')
st.plotly_chart(fig_discount_profit, use_container_width=True)
st.markdown("**Discount above 0.3 often results in negative profit.** 📉")

# Sales and Profit by Category grouped bar
cat_perf = filtered_df.groupby('Category')[['Sales','Profit']].sum().reset_index()
fig_cat_perf = px.bar(cat_perf, x='Category', y=['Sales', 'Profit'], barmode='group', title='💼 Sales & Profit by Category')
st.plotly_chart(fig_cat_perf, use_container_width=True)
st.markdown("**Technology products tend to generate higher sales and profit.** 🚀")

st.markdown("**Monthly sales show seasonal variation – useful for forecasting.** 📊")

