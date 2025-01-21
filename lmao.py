import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Load your dataset
def load_data(file):
    df = pd.read_csv(file)
    return df

# RFM Analysis
def perform_rfm_analysis(df):
    reference_date = pd.Timestamp(datetime.now().date())
    rfm = df.groupby('Customer ID').agg({
        'Purchase Date': lambda x: (reference_date - pd.to_datetime(x).max()).days,
        'Product ID': 'count',
        'Product Discounted Price': 'sum'
    }).rename(columns={
        'Purchase Date': 'Recency',
        'Product ID': 'Frequency',
        'Product Discounted Price': 'Monetary'
    })
    
    quantiles = rfm.quantile(q=[0.25, 0.50, 0.75])
    
    def RScore(x, p, d):
        if p == 'Recency':
            if x <= d[p][0.25]:
                return 4
            elif x <= d[p][0.50]:
                return 3
            elif x <= d[p][0.75]:
                return 2
            else:
                return 1
        else:
            if x <= d[p][0.25]:
                return 1
            elif x <= d[p][0.50]:
                return 2
            elif x <= d[p][0.75]:
                return 3
            else:
                return 4
    
    rfm['R'] = rfm['Recency'].apply(RScore, args=('Recency', quantiles,))
    rfm['F'] = rfm['Frequency'].apply(RScore, args=('Frequency', quantiles,))
    rfm['M'] = rfm['Monetary'].apply(RScore, args=('Monetary', quantiles,))
    
    rfm['Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
    
    rfm['RFM_Customer_segments'] = ''
    rfm.loc[rfm['Score'] >= 9, 'RFM_Customer_segments'] = 'VIP/Loyal'
    rfm.loc[(rfm['Score'] >= 6) & (rfm['Score'] < 9), 'RFM_Customer_segments'] = 'Potential Loyal'
    rfm.loc[(rfm['Score'] >= 5) & (rfm['Score'] < 6), 'RFM_Customer_segments'] = 'At Risk Customers'
    rfm.loc[(rfm['Score'] >= 4) & (rfm['Score'] < 5), 'RFM_Customer_segments'] = 'Can\'t Lose'
    rfm.loc[(rfm['Score'] >= 3) & (rfm['Score'] < 4), 'RFM_Customer_segments'] = 'Lost'
    
    return rfm

# DBSCAN Clustering
def perform_dbscan_clustering(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    rfm['Cluster'] = dbscan.fit_predict(rfm_scaled)
    
    return rfm

# Define marketing strategies based on segments
def recommend_strategies(segment):
    if segment == 'VIP/Loyal':
        return "Offer exclusive discounts, loyalty rewards, and personalized offers."
    elif segment == 'Potential Loyal':
        return "Engage with personalized emails, upsell/cross-sell offers, and loyalty programs."
    elif segment == 'At Risk Customers':
        return "Re-engage with win-back campaigns, special discounts, and personalized communication."
    elif segment == 'Can\'t Lose':
        return "Provide high-value offers, personalized recommendations, and proactive support."
    elif segment == 'Lost':
        return "Re-engage with win-back campaigns, special discounts, and personalized communication."
    else:
        return "No specific strategy defined."

# Streamlit App
def main():
    st.title("Customer Segmentation and Marketing Strategy Recommendation")
    st.write("Upload your dataset to perform RFM analysis and DBSCAN clustering.")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        rfm = perform_rfm_analysis(df)
        rfm = perform_dbscan_clustering(rfm)

        # Display RFM Analysis Results
        st.header("RFM Analysis Results")
        st.write(rfm.head())

        # RFM Segments Distribution
        st.header("RFM Segments Distribution")
        segments_counts = rfm['RFM_Customer_segments'].value_counts().reset_index()
        segments_counts.columns = ['RFM_segment', 'Count']
        fig_rfm = px.bar(segments_counts, x='RFM_segment', y='Count', title='Customer Distribution by RFM Segment')
        st.plotly_chart(fig_rfm)

        # DBSCAN Clusters Visualization
        st.header("DBSCAN Clusters Visualization")
        fig_clusters = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster', title='DBSCAN Clusters')
        st.plotly_chart(fig_clusters)

        # Marketing Strategies
        st.header("Marketing Strategies by Segment")
        strategies = {}
        for segment in rfm['RFM_Customer_segments'].unique():
            strategies[segment] = recommend_strategies(segment)
        
        for segment, strategy in strategies.items():
            st.subheader(f"Segment: {segment}")
            st.write(strategy)

        # Show raw data
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.write(rfm)

# Run the Streamlit app
if __name__ == "__main__":
    main()