import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator import Authenticate
import bcrypt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pickle
import os
import requests

# TMDb API Key (replace with your own key)
TMDB_API_KEY = "7e7e5fed884d3358763c3ffac1cfb245"

# Paths for files
DATASET_PATH = r'C:\Users\PARIMITA\Desktop\Machine learning projects\Movie recommender system\tmdb_5000_movies.csv'
MOVIES_PICKLE_PATH = r'C:\Users\PARIMITA\Desktop\Movie_recommendation_project\movies.pkl'
SIMILARITY_PICKLE_PATH = r'C:\Users\PARIMITA\Desktop\Movie_recommendation_project\similarity.pkl'
GENRE_SIMILARITY_PICKLE_PATH = r'C:\Users\PARIMITA\Desktop\Movie_recommendation_project\genre_similarity.pkl'
CAST_SIMILARITY_PICKLE_PATH = r'C:\Users\PARIMITA\Desktop\Movie_recommendation_project\cast_similarity.pkl'

# # List of plain text passwords
# passwords = ["password123", "mypassword"]
#
# # Hash the passwords
# hashed_passwords = [bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode() for password in passwords]
#
#
# # Credentials setup
# credentials = {
#     "usernames": {
#         "user1": {
#             "name": "John Doe",
#             "password": hashed_passwords[0],
#         },
#         "user2": {
#             "name": "Jane Doe",
#             "password": hashed_passwords[1],
#         },
#     }
# }
#
# # Streamlit authenticator setup
# authenticator = Authenticate(
#     credentials=credentials,
#     cookie_name="movie_recommender_auth",
#     key="movie_recommender_auth_key",
#     cookie_expiry_days=30,
#
# )
#
# # Login section
# name, authentication_status, username = authenticator.login("Login", "main")
#
# if authentication_status:
#     authenticator.logout("Logout", "sidebar")
#     st.sidebar.write(f"Welcome, {name}!")
# elif authentication_status is False:
#     st.error("Username or password is incorrect")
# elif authentication_status is None:
#     st.warning("Please enter your username and password")
#
#




# Function to preprocess data and generate necessary files
def preprocess_and_save():
    movies = pd.read_csv(DATASET_PATH)
    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('[]')
    movies['cast'] = movies['cast'].fillna('[]')  # Include cast here
    movies['vote_average'] = movies['vote_average'].fillna(0.0)  # Ensure vote_average exists
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')  # Convert to datetime
    movies['release_year'] = movies['release_date'].dt.year.fillna(0).astype(int)  # Extract year, fill missing with 0

    # Create a combined cast column (top 5 cast members)
    movies['cast_combined'] = movies['cast'].apply(
        lambda x: " ".join([person['name'] for person in ast.literal_eval(x)[:5]]) if x != '[]' else '')

    # Title-based similarity
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['overview']).toarray()
    similarity = cosine_similarity(vectors)

    # Genre-based similarity
    genre_cv = CountVectorizer(max_features=5000, stop_words='english')
    genre_vectors = genre_cv.fit_transform(movies['genres_combined']).toarray()
    genre_similarity = cosine_similarity(genre_vectors)

    # Cast-based similarity
    cast_cv = CountVectorizer(max_features=5000, stop_words='english')
    cast_vectors = cast_cv.fit_transform(movies['cast_combined']).toarray()
    cast_similarity = cosine_similarity(cast_vectors)

    os.makedirs(os.path.dirname(MOVIES_PICKLE_PATH), exist_ok=True)
    with open(MOVIES_PICKLE_PATH, 'wb') as f:
        pickle.dump(movies[['id', 'title', 'genres', 'genres_combined', 'cast_combined', 'vote_average','release_year']], f)
    with open(SIMILARITY_PICKLE_PATH, 'wb') as f:
        pickle.dump(similarity, f)
    with open(GENRE_SIMILARITY_PICKLE_PATH, 'wb') as f:
        pickle.dump(genre_similarity, f)
    with open(CAST_SIMILARITY_PICKLE_PATH, 'wb') as f:
        pickle.dump(cast_similarity, f)
    print("Files saved successfully.")

# Ensure necessary files are available
if not os.path.exists(MOVIES_PICKLE_PATH) or not os.path.exists(SIMILARITY_PICKLE_PATH) or not os.path.exists(
        GENRE_SIMILARITY_PICKLE_PATH) or not os.path.exists(CAST_SIMILARITY_PICKLE_PATH):
    preprocess_and_save()

# Load necessary files
movies = pickle.load(open(MOVIES_PICKLE_PATH, 'rb'))
similarity = pickle.load(open(SIMILARITY_PICKLE_PATH, 'rb'))
genre_similarity = pickle.load(open(GENRE_SIMILARITY_PICKLE_PATH, 'rb'))
cast_similarity = pickle.load(open(CAST_SIMILARITY_PICKLE_PATH, 'rb'))

# Recommendation functions
def recommend_by_title(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    recommended_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [(movies.iloc[i[0]]['title'], movies.iloc[i[0]]['id']) for i in recommended_indices]
    return recommended_movies

def recommend_by_genre(genre):
    genre = genre.lower()
    genre_indices = [index for index, genres in enumerate(movies['genres_combined']) if genre in genres.lower()]
    if not genre_indices:
        return []
    recommended_movies = []
    for index in genre_indices:
        distances = genre_similarity[index]
        recommended_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies.extend([(movies.iloc[i[0]]['title'], movies.iloc[i[0]]['id']) for i in recommended_indices])
    return list(dict.fromkeys(recommended_movies))[:5]  # Remove duplicates

def recommend_by_cast(cast_name):
    cast_name = cast_name.lower()
    cast_indices = [index for index, cast in enumerate(movies['cast_combined']) if cast_name in cast.lower()]
    if not cast_indices:
        return []
    recommended_movies = []
    for index in cast_indices:
        distances = cast_similarity[index]
        recommended_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies.extend([(movies.iloc[i[0]]['title'], movies.iloc[i[0]]['id']) for i in recommended_indices])
    return list(dict.fromkeys(recommended_movies))[:5]

def recommend_top_vote_average():
    # Get the top 5 movies sorted by vote average in descending order
    top_movies = movies.sort_values(by='vote_average', ascending=False).head(5)
    return [(row['title'], row['id']) for _, row in top_movies.iterrows()]


def recommend_by_release_date(year):
    # Filter movies by release_year
    filtered_movies = movies[movies['release_year'] == year]
    return [(row['title'], row['id']) for _, row in filtered_movies.iterrows()][:5]  # Return top 5 movies

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    return "https://via.placeholder.com/300x450?text=No+Image"

# Streamlit UI
st.title("Movie Recommender System")
option = st.sidebar.radio("Choose Filter Type:", ["Title-Based", "Genre-Based", "Cast-Based", "Vote Average-Based","Release Date-Based"])

if option == "Title-Based":
    selected_movie = st.selectbox("Select a movie:", movies['title'].values)
    if st.button("Get Recommendations"):
        recommendations = recommend_by_title(selected_movie)
        cols = st.columns(len(recommendations))
        for idx, (title, movie_id) in enumerate(recommendations):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{fetch_poster(movie_id)}" style="width: 200px; border-radius: 8px;">
                        <div style="margin-top: 10px;">
                            <span style="font-weight: bold; color: white;">{title}</span>
                        </div>
                        <a href="https://www.themoviedb.org/movie/{movie_id}" 
                           style="text-decoration: none; color: white; background-color: #e50914; 
                                  padding: 8px 16px; border-radius: 5px; margin-top: 15px; display: inline-block;">
                            Watch Now
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

elif option == "Genre-Based":
    genres = set(" ".join(movies['genres_combined']).split())
    selected_genre = st.selectbox("Select a genre:", sorted(genres))
    if st.button("Get Recommendations"):
        recommendations = recommend_by_genre(selected_genre)
        cols = st.columns(len(recommendations))
        for idx, (title, movie_id) in enumerate(recommendations):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{fetch_poster(movie_id)}" style="width: 200px; border-radius: 8px;">
                        <div style="margin-top: 10px;">
                            <span style="font-weight: bold; color: white;">{title}</span>
                        </div>
                        <a href="https://www.themoviedb.org/movie/{movie_id}" 
                           style="text-decoration: none; color: white; background-color: #e50914; 
                                  padding: 8px 16px; border-radius: 5px; margin-top: 15px; display: inline-block;">
                            Watch Now
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

elif option == "Cast-Based":
    cast_names = set(" ".join(movies['cast_combined']).split())
    selected_cast = st.selectbox("Select a cast member:", sorted(cast_names))
    if st.button("Get Recommendations"):
        recommendations = recommend_by_cast(selected_cast)
        cols = st.columns(len(recommendations))
        for idx, (title, movie_id) in enumerate(recommendations):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{fetch_poster(movie_id)}" style="width: 200px; border-radius: 8px;">
                        <div style="margin-top: 10px;">
                            <span style="font-weight: bold; color: white;">{title}</span>
                        </div>
                        <a href="https://www.themoviedb.org/movie/{movie_id}" 
                           style="text-decoration: none; color: white; background-color: #e50914; 
                                  padding: 8px 16px; border-radius: 5px; margin-top: 15px; display: inline-block;">
                            Watch Now
                        </a>
                        <a>log in</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

elif option == "Vote Average-Based":
    if st.button("Get Recommendations"):
        recommendations = recommend_top_vote_average()
        cols = st.columns(len(recommendations))
        for idx, (title, movie_id) in enumerate(recommendations):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{fetch_poster(movie_id)}" style="width: 200px; border-radius: 8px;">
                        <div style="margin-top: 10px;">
                            <span style="font-weight: bold; color: white;">{title}</span>
                        </div>
                        <a href="https://www.themoviedb.org/movie/{movie_id}" 
                           style="text-decoration: none; color: white; background-color: #e50914; 
                                  padding: 8px 16px; border-radius: 5px; margin-top: 15px; display: inline-block;">
                            Watch Now
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
elif option == "Release Date-Based":
    release_date = st.date_input("Enter Release Date:", value=pd.to_datetime("2020-01-01"))
    if st.button("Get Recommendations"):
        recommendations = recommend_by_release_date(release_date)

        if recommendations:
            cols = st.columns(len(recommendations))
            for idx, (title, movie_id) in enumerate(recommendations):
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="{fetch_poster(movie_id)}" style="width: 200px; border-radius: 8px;">
                            <div style="margin-top: 10px;">
                                <span style="font-weight: bold; color: white;">{title} ({release_date})</span>
                            </div>
                            <a href="https://www.themoviedb.org/movie/{movie_id}" 
                               style="text-decoration: none; color: white; background-color: #e50914; 
                                      padding: 8px 16px; border-radius: 5px; margin-top: 15px; display: inline-block;">
                                Watch Now
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.warning(f"No movies found for the date {release_date}. Please try a different year.")

