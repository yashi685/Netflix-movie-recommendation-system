import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import ast
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Movie Show Clustering", layout="wide")
st.title("🎬 Netflix-style Movie Clustering using KMeans")
st.markdown("Group similar movies based on **Genre**, **Rating**, and **Duration** using KMeans clustering.")

uploaded_file = st.file_uploader("C:\\Users\\yashi\\OneDrive\\Desktop\\netflix_recommendationsystem\\movies_metadata.csv", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file, low_memory=False)
    
    # Select relevant columns
    df = df[['title', 'genres', 'vote_average', 'runtime']]
    
    # Drop missing or invalid data
    df = df[pd.to_numeric(df['runtime'], errors='coerce').notnull()]
    df['runtime'] = df['runtime'].astype(float)
    df = df[df['vote_average'].notnull()]
    
    # Process genres
    def extract_genres(x):
        try:
            genres = ast.literal_eval(x)
            return [d['name'] for d in genres]
        except:
            return []
    
    df['genres'] = df['genres'].apply(extract_genres)
    
    # One-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
    
    # Combine features
    df_cluster = pd.concat([df[['title', 'vote_average', 'runtime']], genre_encoded], axis=1)
    
    # Scale features
    features = df_cluster.drop(['title'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Sidebar: choose number of clusters
    st.sidebar.header("⚙️ Clustering Settings")
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_cluster['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Visualize cluster counts
    st.subheader("📊 Cluster Distribution")
    st.bar_chart(df_cluster['cluster'].value_counts().sort_index())
    
    # Scatter plot
    st.subheader("📍 Cluster Visualization (Rating vs Runtime)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_cluster, x='vote_average', y='runtime', hue='cluster', palette='tab10', s=60, ax=ax)
    ax.set_title("Movie Clusters by Rating and Duration")
    st.pyplot(fig)
    
    # Show cluster details
    st.subheader("🔍 Explore Movies by Cluster")
    selected_cluster = st.selectbox("Choose a cluster", sorted(df_cluster['cluster'].unique()))
    st.dataframe(df_cluster[df_cluster['cluster'] == selected_cluster][['title', 'vote_average', 'runtime']].head(20))
    
    # Download button
    csv_data = df_cluster.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Clustered Data", csv_data, file_name="clustered_movies.csv", mime="text/csv")
    
else:
    st.info("👈 Please upload the `movies_metadata.csv` file to begin.")
