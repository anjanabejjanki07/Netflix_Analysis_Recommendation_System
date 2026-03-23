
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df = df.dropna(subset=["title"])
    df["listed_in"] = df["listed_in"].fillna("")
    df["description"] = df["description"].fillna("")
    df["country"] = df["country"].fillna("Unknown")
    return df

df = load_data()

# ------------------ TITLE ------------------
st.title("🎬 Netflix Interactive Analytics & Recommendation System")

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔍 Filter Options")

# Genre list
all_genres = df['listed_in'].str.split(', ').explode().unique()
genre = st.sidebar.selectbox("Select Genre", ["All"] + sorted(all_genres))

# Country list
country = st.sidebar.selectbox("Select Country", ["All"] + sorted(df['country'].unique()))

# Type filter
type_filter = st.sidebar.selectbox("Type", ["All"] + list(df['type'].unique()))

# Apply filters
filtered_df = df.copy()

if genre != "All":
    filtered_df = filtered_df[filtered_df['listed_in'].str.contains(genre)]

if country != "All":
    filtered_df = filtered_df[filtered_df['country'] == country]

if type_filter != "All":
    filtered_df = filtered_df[filtered_df['type'] == type_filter]

# ------------------ SEARCH ------------------
st.subheader("🔎 Search Movie / Show")
search_query = st.text_input("Enter movie or show name")

if search_query:
    search_result = df[df['title'].str.contains(search_query, case=False)]
    st.write(search_result[['title', 'type', 'country', 'listed_in']])

# ------------------ SHOW DATA ------------------
st.subheader("📋 Filtered Results")
st.dataframe(filtered_df[['title', 'type', 'country', 'listed_in']].head(50))

# ------------------ VISUALIZATIONS ------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

# Top Genres
with col1:
    st.write("Top 10 Genres")
    genre_count = df['listed_in'].str.split(', ').explode().value_counts().head(10)
    st.bar_chart(genre_count)

# Top Countries
with col2:
    st.write("Top 10 Countries")
    country_count = df['country'].value_counts().head(10)
    st.bar_chart(country_count)

# ------------------ RECOMMENDATION SYSTEM ------------------
st.subheader("🤖 Movie Recommendation System")

# Combine features
df['combined'] = df['listed_in'] + " " + df['description']

# Vectorization
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(df['combined'])

# Similarity
similarity = cosine_similarity(matrix)

# Dropdown for movie selection
movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Select a Movie", movie_list)

# Recommendation function
def recommend(movie):
    try:
        index = df[df['title'] == movie].index[0]
        distances = list(enumerate(similarity[index]))
        movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

        recommended_titles = []
        for i in movies:
            recommended_titles.append(df.iloc[i[0]].title)

        return recommended_titles
    except:
        return []

# Button
if st.button("Get Recommendations"):
    results = recommend(selected_movie)

    if results:
        st.write("### 🎯 Recommended Movies:")
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")
    else:
        st.write("No recommendations found.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.write("Made with ❤️ using Streamlit")