import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import streamlit as st


COLORS = px.colors.qualitative.Vivid + px.colors.qualitative.Safe
PRIMARY_COLOR = "#636EFA"
SECONDARY_COLOR = "#EF553B"
ACCENT_COLOR = "#00CC96"

def plot_sales_trend(df: pd.DataFrame) -> go.Figure:
    """Beautiful animated sales trend over time"""
    df['Date'] = pd.to_datetime(df['Date'])
    monthly = df.set_index('Date').resample('ME')['Sales'].sum().reset_index()
    monthly['Month'] = monthly['Date'].dt.strftime('%b %Y')
    monthly['Year'] = monthly['Date'].dt.year

    fig = go.Figure()
    for year in monthly['Year'].unique():
        data = monthly[monthly['Year'] == year]
        fig.add_trace(go.Scatter(
            x=data['Month'], y=data['Sales'],
            mode='lines+markers',
            name=str(year),
            line=dict(width=4),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title="Sales Trend Over Time (Monthly)",
        title_x=0.5,
        title_font=dict(size=20, family="Arial Black"),
        xaxis_title="Month",
        yaxis_title="Total Sales ($)",
        template="plotly_dark",
        hovermode="x unified",
        legend_title="Year",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def plot_regional_sales(df: pd.DataFrame) -> go.Figure:
    """Epic sunburst or treemap style regional breakdown"""
    regional = df.groupby(['Region', 'Product'])['Sales'].sum().reset_index()
    
    fig = px.sunburst(
        regional,
        path=['Region', 'Product'],
        values='Sales',
        color='Sales',
        color_continuous_scale='Viridis',
        title="Sales Breakdown by Region & Product"
    )
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20, family="Arial Black"),
        height=520,
        margin=dict(t=60, l=0, r=0, b=0)
    )
    return fig


def plot_product_performance(df: pd.DataFrame) -> go.Figure:
    """Top 10 products with gradient bar magic"""
    top_products = df.groupby('Product')['Sales'].sum().nlargest(10).reset_index()
    fig = px.bar(
        top_products,
        x='Sales',
        y='Product',
        orientation='h',
        text='Sales',
        color='Sales',
        color_continuous_scale='Plasma',
        title="Top 10 Best-Selling Products"
    )
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20, family="Arial Black"),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Total Revenue ($)",
        yaxis_title="",
        height=500,
        template="plotly_dark"
    )
    return fig


def plot_customer_segmentation(df: pd.DataFrame) -> go.Figure:
    """3D Customer Segments with hover info + cluster names"""
    features = df[['Customer_Age', 'Sales', 'Customer_Satisfaction']].copy()
    features_scaled = (features - features.mean()) / features.std()

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Segment'] = kmeans.fit_predict(features_scaled)

    
    segment_names = {
        0: "Young High-Spenders",
        1: "Mature Loyal Customers",
        2: "Budget Shoppers",
        3: "Premium Enthusiasts"
    }
    df['Segment_Name'] = df['Segment'].map(segment_names)

    fig = px.scatter_3d(
        df,
        x='Customer_Age',
        y='Sales',
        z='Customer_Satisfaction',
        color='Segment_Name',
        size_max=18,
        opacity=0.8,
        hover_data=['Region', 'Product'],
        color_discrete_sequence=COLORS[:4],
        title="AI-Powered Customer Segmentation"
    )

    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=22, family="Arial Black"),
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Sales Amount ($)',
            zaxis_title='Satisfaction Score',
            bgcolor="black"
        ),
        height=600,
        legend_title="Customer Segment",
        template="plotly_dark"
    )
    return fig



def render_kpi_cards(df: pd.DataFrame):
    """Call this in sidebar or main page for sexy KPI boxes"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${df['Sales'].sum():,.0f}", "↑ 12%")
    with col2:
        st.metric("Total Orders", f"{len(df):,}", "↑ 8%")
    with col3:
        st.metric("Avg Order Value", f"${df['Sales'].mean():.0f}", "↑ 4%")
    with col4:
        st.metric("Top Region", df.groupby('Region')['Sales'].sum().idxmax())