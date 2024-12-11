# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the perfume dataset
def load_data():
    try:
        return pd.read_csv("final_perfume_data.csv", encoding='unicode_escape')
    except FileNotFoundError:
        st.write("Dataset file not found!")
        return pd.DataFrame()

# Preprocess the data
def preprocess_data(df):
    df = df.drop(columns=["Image URL"], errors="ignore")  # Drop irrelevant columns
    df['Combined_Features'] = df['Description'].fillna('') + ' ' + df['Notes'].fillna('')
    return df

# Compute cosine similarity matrix
def compute_similarity(df):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(df['Combined_Features'])
    return cosine_similarity(count_matrix, count_matrix)

# Recommendation function
def recommend_perfumes(perfume_name, num_recommendations, df, cosine_sim):
    perfume_idx = None

    for idx, name in enumerate(df['Name']):
        if perfume_name.lower() in str(name).lower():
            perfume_idx = idx
            break

    if perfume_idx is None:
        return None

    sim_scores = [(i, score) for i, score in enumerate(cosine_sim[perfume_idx])]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]

    recommended_indices = [idx for idx, _ in sim_scores]
    return df.iloc[recommended_indices][['Name', 'Brand']]

# Load and preprocess the data
data = load_data()

if not data.empty:
    data = preprocess_data(data)
    cosine_sim = compute_similarity(data)

    # Streamlit Interface
    st.title("Scentsible Recommendations")
    st.markdown(
        "Discover your next favorite scent with our AI-powered recommender system. Simply enter a perfume name to see similar recommendations! Let's make it make scents!"
    )

    # User input for perfume name
    st.subheader("Input Perfume Name")
    perfume_name = st.text_input("Enter a perfume name:")

    # Number of recommendations
    num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

    # Button to get recommendations
    if st.button("Recommend"):
        if perfume_name:
            recommendations = recommend_perfumes(perfume_name, num_recommendations, data, cosine_sim)
            if recommendations is not None:
                st.subheader("Recommended Perfumes")
                st.dataframe(recommendations)
            else:
                st.error(f"No perfume found with the name '{perfume_name}'.")
        else:
            st.warning("Please enter a perfume name.")

else:
    st.write("No data available to make recommendations.")
