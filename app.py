import streamlit as st
import pandas as pd
import sqlite3
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


import base64

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("hobby5.png")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif !important;
    color: #003366 !important;
}

h1, h2, h3 {
    font-family: 'Montserrat', sans-serif !important;
    color: #003366 !important;
    font-weight: 600;
}

label, button {
    font-family: 'Montserrat', sans-serif !important;
    color: #003366 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Glass effect sidebar */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.3);
}

/* Text style */
[data-testid="stSidebar"] * {
    font-family: 'Poppins', sans-serif;
    color: #003366 !important;
}

/* Hover effect */
[data-testid="stSidebar"] div:hover {
    background: rgba(0, 51, 102, 0.1);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='
    text-align: center;
    color: #003366;
    font-family: Montserrat;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
'>
Predictive Model for Student Hobby Analysis
</h1>
""", unsafe_allow_html=True)



# =========================
# DATABASE
# =========================

conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
username TEXT,
password TEXT
)
""")
conn.commit()


def register_user(username, password):
    c.execute("INSERT INTO users(username,password) VALUES(?,?)",
              (username, password))
    conn.commit()


def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, password))
    return c.fetchone()


# =========================
# SESSION STATE
# =========================

if "login" not in st.session_state:
    st.session_state.login = False

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "encoder" not in st.session_state:
    st.session_state.encoder = None
    
if "username" not in st.session_state:
    st.session_state.username = ""
    
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "answered" not in st.session_state:
    st.session_state.answered = False

if "page" not in st.session_state:
    st.session_state.page = None
    
if "feedback_page" not in st.session_state:
    st.session_state.feedback_page = False

# =========================
# LOGIN / REGISTER
# =========================

if not st.session_state.login:

    menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

    if menu == "Register":
        st.title("Register")

        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Register"):
            register_user(user, pwd)
            st.success("Registered Successfully")

    elif menu == "Login":
        st.title("Login")

        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(user, pwd):
                st.session_state.login = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error("Invalid Credentials")

# =========================
# MAIN APP
# =========================

else:

    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    
    menu = st.sidebar.selectbox(
        "Menu",
        [
            "Upload Dataset",
            "View Dataset",
            "Data Cleaning",
            "Train Model",
            "Load Model",
            "Prediction"
        ]
        
    )
    
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Brain Booster")

    if st.sidebar.button("Puzzles & Games"):
        st.session_state.page = "games"
    if st.sidebar.button("Feedback"):
        st.session_state.page = "feedback"   
   # =========================
# FEEDBACK PAGE
# =========================
if st.session_state.page=="feedback":

    st.title("📝 Feedback Section")

    feedback = st.text_area("Write your feedback about the app:")

    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

    st.stop()
# =========================
# PUZZLES & GAMES PAGE
# =========================


elif st.session_state.page == "games":

    st.title("🎮 Hobby & Student Life Puzzle Zone")

    game_type = st.selectbox(
        "Choose Game Type",
        ["MCQ Quiz", "Statistical Puzzles"]
    )

    # =========================
    # MCQ QUIZ
    # =========================
    if game_type == "MCQ Quiz":

        quiz_questions = [
            {
                "question": "A student spends most time drawing and designing. What hobby is likely?",
                "options": ["Sports", "Arts", "Gaming", "Reading"],
                "answer": "Arts"
            },
            {
                "question": "If a student practices cricket daily, what hobby category is this?",
                "options": ["Arts", "Sports", "Music", "Coding"],
                "answer": "Sports"
            },
            {
                "question": "Which activity improves logical thinking the most?",
                "options": ["Painting", "Chess", "Dancing", "Singing"],
                "answer": "Chess"
            }
        ]

        q_index = st.session_state.quiz_index
        q = quiz_questions[q_index]

        st.subheader(f"Q{q_index + 1}: {q['question']}")

        selected = st.radio("Choose answer:", q["options"], key=f"q_{q_index}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Submit Answer"):
                if not st.session_state.answered:

                    if selected == q["answer"]:
                        st.success("Correct Answer ✅")
                        st.session_state.quiz_score += 1
                    else:
                        st.error(f"Wrong ❌ | Correct Answer: {q['answer']}")

                    st.session_state.answered = True

        with col2:
            if st.session_state.answered:
                if st.button("Next Question"):
                    st.session_state.quiz_index += 1
                    st.session_state.answered = False
                    st.rerun()

        st.sidebar.write("🎯 Score:", st.session_state.quiz_score)


    # =========================
    # STATISTICAL PUZZLES
    # =========================
    elif game_type == "Statistical Puzzles":

        import random

        puzzle_type = st.selectbox(
            "Choose Statistical Puzzle",
            [
                "Mean",
                "Probability",
                "Percentage",
                "Mode",
                "Data Interpretation",
                "Median",
                "Thinking"
            ]
        )

        st.markdown("---")

        # MEAN
        if puzzle_type == "Mean":
            q = random.choice([
                {"q": "Mean of 10,20,30,40?", "a": 25},
                {"q": "Mean of 5,15,25?", "a": 15}
                
            ])

            st.write(q["q"])
            ans = st.number_input("Answer", key="mean")

            if st.button("Check"):
                if ans == q["a"]:
                    st.success("Correct 📊")
                else:
                    st.error(f"Wrong ❌ {q['a']}")

        # PROBABILITY
        elif puzzle_type == "Probability":
            q = {"q": "3 red, 2 blue balls. Probability red?", "a": "3/5"}

            st.write(q["q"])
            ans = st.radio("Choose:", ["3/5", "2/5", "1/2", "1/6"])

            if st.button("Check"):
                if ans == q["a"]:
                    st.success("Correct 🎯")
                else:
                    st.error(f"Wrong ❌ {q['a']}")

        # PERCENTAGE
        elif puzzle_type == "Percentage":
            q = {"q": "20% of 150?", "a": 30}

            st.write(q["q"])
            ans = st.number_input("Answer", key="percent",min_value=1, max_value=50, step=1)

            if st.button("Check"):
                if ans == q["a"]:
                    st.success("Correct 📈")
                else:
                    st.error(f"Wrong ❌ {q['a']}")

        # MODE
        elif puzzle_type == "Mode":
            q = {"q": "Find mode: 2,3,3,5,7", "a": 3}

            st.write(q["q"])
            ans = st.number_input("Answer", key="mode",min_value=1, max_value=50, step=1)

            if st.button("Check"):
                if ans == q["a"]:
                    st.success("Correct 📊")
                else:
                    st.error(f"Wrong ❌ {q['a']}")

        # DATA
        elif puzzle_type == "Data Interpretation":
            q = {"q": "Arts:10, Sports:15, Music:5 → Most popular?", "a": "Sports"}

            st.write(q["q"])
            ans = st.selectbox("Answer", ["Arts", "Sports", "Music"])

            if st.button("Check"):
                if ans == q["a"]:
                    st.success("Correct 📊")
                else:
                    st.error(f"Wrong ❌ {q['a']}")

        # MEDIAN
        elif puzzle_type == "Median":
            q = {"q": "Median of 5,10,15?", "a": 10}

            st.write(q["q"])
            ans = st.number_input("Answer", key="Median",min_value=1, max_value=50, step=1)

            if st.button("Check"):
                if ans == q["a"]:
                    st.success("Correct 📊")
                else:
                    st.error(f"Wrong ❌ {q['a']}")

        # THINKING
        elif puzzle_type == "Thinking":
            q = {"q": "Best for analysis?", "a": "Large dataset"}

            st.write(q["q"])
            ans = st.radio("Choose:", ["One value", "Large dataset", "Random guess", "Opinion"])

            if st.button("Check"):
                if ans == q["a"]:
                    st.success("Correct 🧠")
                else:
                    st.error(f"Wrong ❌ {q['a']}")
       
                          
    
elif st.session_state.page is None:
    
    # =========================
    # UPLOAD
    # =========================

    if menu == "Upload Dataset":

        st.title("Upload Dataset")

        file = st.file_uploader("Hobby_Data.csv", type=["csv"])

        if file:
            df = pd.read_csv(file)
            st.session_state.df = df

            st.success("Dataset Loaded")
            st.dataframe(df)

    # =========================
    # VIEW
    # =========================

    elif menu == "View Dataset":

        st.title("Dataset")

        if st.session_state.df is not None:
            st.subheader("First 10 Rows")
            st.dataframe(st.session_state.df.head(10))
            
            st.subheader("Complete Dataset")
            st.dataframe(st.session_state.df)
        else:
            st.warning("Upload dataset first")

    # =========================
    # CLEANING + ENCODING
    # =========================

   elif menu == "Data Cleaning":

        st.title("Data Cleaning")

        df = st.session_state.df

        if df is None:
            st.warning("Upload dataset first")

        else:

            st.write("Column names:", list(df.columns))
            st.write("Column types:", df.dtypes)

            st.write("Missing Values")
            st.write(df.isnull().sum())

            st.write("Duplicates:", df.duplicated().sum())

            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.session_state.df = df
                st.success("Removed")

            if st.button("Apply Encoding"):

                target_col = "Predicted Hobby"

                if target_col not in df.columns:
                    st.error(f"Column '{target_col}' not found! Actual columns: {list(df.columns)}")
                else:
                    feature_cols = [c for c in df.columns if c != target_col]

                    for col in feature_cols:
                        if df[col].dtype == "object":
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))

                    target_encoder = LabelEncoder()
                    df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))

                    joblib.dump(target_encoder, "target_encoder.pkl")

                    st.session_state.df = df

                    st.success("Encoding Done")
                    st.dataframe(df.head())

    # =========================
    # TRAIN MODEL
    # =========================

    elif menu == "Train Model":

        st.title("Train Model")

        df = st.session_state.df

        if df is None:
            st.warning("Upload dataset first")

        else:

            target_col = st.selectbox("Select Target Column", df.columns)

            if st.button("Train the Model"):

                X = df.drop(target_col, axis=1)
                y = df[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)

                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                joblib.dump(model, "hobby.pkl")

                st.success("Model Trained Successfully")
                st.write("Accuracy:", round(acc * 100, 2), "%")

    # =========================
    # LOAD MODEL (NEW MENU)
    # =========================

    elif menu == "Load Model":

        st.title("Load Saved Model")

        try:
            st.session_state.model = joblib.load("hobby.pkl")
            st.session_state.encoder = joblib.load("target_encoder.pkl")

            st.success("Model and Encoder Loaded Successfully")

        except:
            st.error("No saved model found. Train first.")

    # =========================
    # PREDICTION
    # =========================

    elif menu == "Prediction":

        st.title("Predict Hobby")

        model = st.session_state.model
        encoder = st.session_state.encoder

        if model is None or encoder is None:
            st.warning("Please load model first from 'Load Model'")
            st.stop()

        Olympiad_Participation = st.selectbox("Olympiad Participation", [0, 1])
        Scholarship = st.selectbox("Scholarship", [0, 1])
        School = st.selectbox("School ", [0, 1])
        Fav_sub = st.number_input("Favourite Subject (Encoded Value)", min_value=0, step=1)

        Projects = st.selectbox("Projects", [0, 1])
        Grasp_pow = st.number_input("Grasp Percentage (1-10)", min_value=1, max_value=10, step=1)
        Time_sprt = st.number_input("Time Spent on Sports", min_value=0)

        Medals = st.selectbox("Medals", [0, 1])
        Career_sprt = st.selectbox("Career Support", [0, 1])
        Act_sprt = st.selectbox("Activity Support", [0, 1])

        Fant_arts = st.selectbox("Interested in Arts", [0, 1])
        Won_arts = st.selectbox("Won Arts Competition", [0, 1])
        Time_art = st.number_input("Time Spent on Arts", min_value=0)


        if st.button("Predict"):

            try:
                data = [[
                    Olympiad_Participation,
                Scholarship,
                School,
                Fav_sub,
                Projects,
                Grasp_pow,
                Time_sprt,
                Medals,
                Career_sprt,
                Act_sprt,
                Fant_arts,
                Won_arts,
                Time_art
                ]]

                pred = model.predict(data)
                result = encoder.inverse_transform(pred)[0]

                st.success(f"Predicted Hobby: {result}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
    