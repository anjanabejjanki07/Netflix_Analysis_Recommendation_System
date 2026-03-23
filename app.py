import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

st.title("🎬 Netflix Interactive Analytics & Recommendation System")

# ------------------ DEFAULT SAMPLE DATA ------------------
@st.cache_data
def load_sample_data():
    data = {
        "title": ["Stranger Things", "Money Heist", "Breaking Bad", "The Witcher", "Narcos"],
        "type": ["TV Show", "TV Show", "TV Show", "TV Show", "TV Show"],
        "country": ["USA", "Spain", "USA", "Poland", "USA"],
        "listed_in": ["Drama, Sci-Fi", "Crime, Thriller", "Crime, Drama", "Fantasy, Action", "Crime, Drama"],
        "description": [
            "A group of kids face supernatural forces.",
            "A group plans the biggest heist in history.",
            "A chemistry teacher turns into a drug lord.",
            "A monster hunter struggles in a magical world.",
            "Story of drug cartels in Colombia."
        ]
    }
    return pd.DataFrame(data)

# ------------------ FILE UPLOAD ------------------
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Netflix CSV File", type=["csv"])

# ------------------ DATA SELECTION ------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded dataset")
else:
    df = load_sample_data()
    st.info("ℹ️ Showing sample dataset (Upload your own for full analysis)")

# ------------------ CLEAN DATA ------------------
df = df.dropna(subset=["title"])
df["listed_in"] = df["listed_in"].fillna("")
df["description"] = df["description"].fillna("")
df["country"] = df["country"].fillna("Unknown")

# ------------------ SIDEBAR FILTERS ------------------
st.sidebar.header("🔍 Filter Options")

all_genres = df['listed_in'].str.split(', ').explode().unique()
genre = st.sidebar.selectbox("Select Genre", ["All"] + sorted(all_genres))

country = st.sidebar.selectbox("Select Country", ["All"] + sorted(df['country'].unique()))
type_filter = st.sidebar.selectbox("Type", ["All"] + list(df['type'].unique()))

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
st.dataframe(filtered_df[['title', 'type', 'country', 'listed_in']])

# ------------------ VISUALIZATION ------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Top Genres")
    genre_count = df['listed_in'].str.split(', ').explode().value_counts()
    st.bar_chart(genre_count)

with col2:
    st.write("Top Countries")
    country_count = df['country'].value_counts()
    st.bar_chart(country_count)

# ------------------ RECOMMENDATION SYSTEM ------------------
st.subheader("🤖 Movie Recommendation System")

df['combined'] = df['listed_in'] + " " + df['description']

cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(df['combined'])

similarity = cosine_similarity(matrix)

movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Select a Movie", movie_list)

def recommend(movie):
    try:
        index = df[df['title'] == movie].index[0]
        distances = list(enumerate(similarity[index]))
        movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
        return [df.iloc[i[0]].title for i in movies]
    except:
        return []

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
