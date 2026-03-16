import streamlit as st
import pandas as pd
import math

# --- DATA PREPARATION ---
# Survey data collected regarding movie preferences and viewing decisions
data = {
    'Genre': ['Drama', 'Drama', 'Horror', 'Horror', 'Horror', 'Action', 'Drama', 'Action', 'Drama', 'Action'],
    'Runtime': ['Short', 'Short', 'Long', 'Long', 'Long', 'Long', 'Short', 'Long', 'Long', 'Short'],
    'Friends': ['Without', 'Without', 'With', 'Without', 'With', 'Without', 'Without', 'Without', 'With', 'Without'],
    'Mood': ['Happy', 'Happy', 'Happy', 'Tired', 'Tired', 'Happy', 'Sad', 'Happy', 'Sad', 'Tired'],
    'Watch': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

def calculate_naive_bayes(genre, runtime, friends, mood):
    """
    Implements the Naive Bayes formula: P(Y|X) = P(Y) * P(X1|Y) * P(X2|Y) * ...
    """
    classes = ['Yes', 'No']
    results = {}
    
    for cls in classes:
        # Prior Probability: P(Y)
        prior = len(df[df['Watch'] == cls]) / len(df)
        
        # Likelihoods: P(Xi|Y) 
        # Using Laplace Smoothing to handle potential zero-frequency issues
        subset = df[df['Watch'] == cls]
        p_genre = (len(subset[subset['Genre'] == genre]) + 1) / (len(subset) + df['Genre'].nunique())
        p_runtime = (len(subset[subset['Runtime'] == runtime]) + 1) / (len(subset) + df['Runtime'].nunique())
        p_friends = (len(subset[subset['Friends'] == friends]) + 1) / (len(subset) + df['Friends'].nunique())
        p_mood = (len(subset[subset['Mood'] == mood]) + 1) / (len(subset) + df['Mood'].nunique())
        
        # Posterior Calculation
        posterior = prior * p_genre * p_runtime * p_friends * p_mood
        results[cls] = posterior
        
    return results

# --- STREAMLIT UI ---
st.set_page_config(page_title="Movie Decision Predictor", page_icon="🎬")

st.title("🎬 Movie Viewing Predictor")
st.markdown("Predicts if a person will watch a movie based on Naive Bayes analysis of survey data.")

# Sidebar - Dataset Statistics
st.sidebar.header("Survey Statistics")
st.sidebar.write(f"Total Responses: {len(df)}")
st.sidebar.write(f"Total 'Yes' Decisions: {len(df[df['Watch']=='Yes'])}")
st.sidebar.write(f"Total 'No' Decisions: {len(df[df['Watch']=='No'])}")
st.sidebar.table(df)

# Input Section
st.subheader("Select Conditions")
col1, col2 = st.columns(2)

with col1:
    genre_in = st.selectbox("Genre", df['Genre'].unique())
    runtime_in = st.selectbox("Runtime", df['Runtime'].unique())

with col2:
    friends_in = st.selectbox("Friends", df['Friends'].unique())
    mood_in = st.selectbox("Current Mood", df['Mood'].unique())

if st.button("Predict Decision", type="primary"):
    probs = calculate_naive_bayes(genre_in, runtime_in, friends_in, mood_in)
    prediction = max(probs, key=probs.get)
    
    st.divider()
    
    # Display Result
    if prediction == "Yes":
        st.success(f"### PREDICTION: The person WILL watch the movie.")
    else:
        st.error(f"### PREDICTION: The person will NOT watch the movie.")
        
    # Technical Probability Breakdown
    st.subheader("Probability Breakdown")
    total_prob = sum(probs.values())
    for cls, val in probs.items():
        percentage = (val / total_prob) * 100
        st.write(f"**Score for '{cls}':** {val:.6f} ({percentage:.1f}%)")

    st.latex(r"P(Y \mid X) = P(Y) \prod_{i=1}^{n} P(X_i \mid Y)")