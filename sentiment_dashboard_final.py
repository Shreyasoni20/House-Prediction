# sentiment_dashboard_final.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import joblib
import speech_recognition as sr
import numpy as np
import os
import cv2
import time
from textblob import TextBlob 
import re

# ---------------- SENTIMENT FIX ----------------
def fix_sentiment(text, pred_label):
    text = text.lower()

    # 🔴 Negation handling
    negative_patterns = [
        "not good", "not feeling good", "not happy",
        "not satisfied", "not nice", "not great",
        "not working", "not okay", "bad", "worst",
        "not feeling well", "not fine", "not useful"
    ]

    for pattern in negative_patterns:
        if pattern in text:
            return "Negative"

    # 🟣 Sarcasm handling
    sarcastic_patterns = [
        "yeah right", "very nice", "great job",
        "wow amazing", "just perfect"
    ]

    for pattern in sarcastic_patterns:
        if pattern in text:
            return "Sarcastic"

    # ⚡ Strong negation rule
    if "not" in text and ("good" in text or "happy" in text):
        return "Negative"

    return pred_label

# ---------------- CONFIG ----------------
MODEL_PATH = r"c:\Users\user\OneDrive\python\analyzer.pkl"
VECT_PATH = r"c:\Users\user\OneDrive\python\insight.pkl"

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "200805",
    "database": "e_consultancy"
}

COLOR_MAP = {
    "Positive": "#2ecc71",
    "Neutral": "#f39c12",
    "Negative": "#e74c3c",
    "Sarcastic": "#9b59b6"
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model_and_vectorizer(model_path=MODEL_PATH, vect_path=VECT_PATH):
    model, vect, errors = None, None, []
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            errors.append(f"Model not found: {model_path}")
    except Exception as e:
        errors.append(f"Model load error: {e}")
    try:
        if os.path.exists(vect_path):
            vect = joblib.load(vect_path)
        else:
            errors.append(f"Vectorizer not found: {vect_path}")
    except Exception as e:
        errors.append(f"Vectorizer load error: {e}")
    return model, vect, errors


model, vectorizer, load_errors = load_model_and_vectorizer()
if load_errors:
    for e in load_errors:
        st.sidebar.error(e)


# ---------------- DB helpers ----------------
def create_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        st.error(f"MySQL connection error: {e}")
    return None


def init_db():
    conn = create_connection()
    if conn:
        cur = conn.cursor()
        cur.execute(f"""CREATE TABLE IF NOT EXISTS consultations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            serial_no VARCHAR(100),
            category VARCHAR(100),
            comment TEXT,
            sentiment VARCHAR(20),
            confidence FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        cur.close()
        conn.close()


def insert_record(serial_no, category, comment, sentiment, confidence):
    conn = create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO consultations 
                     (serial_no, category, comment, sentiment, confidence, timestamp) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (serial_no, category, comment, sentiment, confidence, datetime.now()))
            conn.commit()
        except Error as e:
            st.error(f"Insert failed: {e}")
        finally:
            cursor.close()
            conn.close()


init_db()

# ---------------- FACE RECOGNITION ----------------
DATASET_PATH = r"C:\Users\user\Desktop\data"

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = [item[1] for item in dk]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]


def run_face_recognition():
    face_data, label = [], []
    class_id, names = 0, {}

    for fx in os.listdir(DATASET_PATH):
        if fx.endswith('.npy'):
            names[class_id] = fx[:-4]
            data_item = np.load(os.path.join(DATASET_PATH, fx))
            face_data.append(data_item)
            target = class_id * np.ones((data_item.shape[0],))
            label.append(target)
            class_id += 1

    if len(face_data) == 0:
        st.warning("⚠️ No face data found in dataset folder.")
        return

    face_dataset = np.concatenate(face_data, axis=0)
    face_label = np.concatenate(label, axis=0).reshape((-1, 1))
    trainset = np.concatenate((face_dataset, face_label), axis=1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("🚫 Unable to access camera. Check permissions or if it's already in use.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Frame not captured. Try restarting camera.")
            break
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            face_section = frame[y:y+h, x:x+w]
            face_section = cv2.resize(face_section, (100, 100))
            out = knn(trainset, face_section.flatten())
            pred_name = names.get(int(out), "Unknown")
            cv2.putText(frame, pred_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow("Face Recognition", frame)
        if time.time() - start_time > 10 or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------------- STYLE ----------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Cinzel+Decorative:wght@700&family=Dancing+Script:wght@700&display=swap" rel="stylesheet">

<style>
.stApp { background-color: #faf6f2; font-family: 'Poppins', sans-serif !important; }
[data-testid="stTabs"] button {
    background-color: #4A2C2A !important;
    color: white !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    font-size: 18px !important;
    margin-right: 5px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] { border-bottom: 4px solid #E0B37D !important; }

.title { font-family: 'Cinzel Decorative', serif; font-size: 70px; font-weight: 700; color: #4A2C2A; text-align: center; }
.subtitle { text-align:center; color:#593C39; font-size:28px; }

@keyframes moveText { 0% { transform: translateX(100%); } 100% { transform: translateX(-100%); } }
.moving-text { width: 100%; overflow: hidden; white-space: nowrap; color: #4A2C2A; font-size: 22px; margin-top: 10px; }
.moving-text span { display: inline-block; padding-left: 100%; animation: moveText 20s linear infinite; }

.carousel-container { position: relative; width: 90%; height: 450px; margin: 0 auto; overflow: hidden; border-radius: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
.carousel-image { width: 100%; height: 450px; object-fit: cover; border-radius: 16px; position: absolute; animation: fade 15s infinite; }
.carousel-image:nth-child(1){animation-delay:0s;} .carousel-image:nth-child(2){animation-delay:5s;} .carousel-image:nth-child(3){animation-delay:10s;}
@keyframes fade { 0%{opacity:0;}10%{opacity:1;}33%{opacity:1;}43%{opacity:0;}100%{opacity:0;} }

.info-card { background:white; padding:30px; border-radius:16px; box-shadow:4px 4px 12px rgba(0,0,0,0.08); border:2px solid #e3c49b; text-align:center; }

.metric-card { background:white; border-radius:14px; padding:24px; box-shadow:0 6px 18px rgba(0,0,0,0.06); border:2px solid #e9d5bf; text-align:center; }
.metric-card h4 { margin:0; font-family:'Dancing Script', cursive; color:#4b2b28; font-size:28px; }
.metric-value { font-size:36px; font-weight:800; color:#4b2b28; font-family:'Poppins'; }
</style>
""", unsafe_allow_html=True)


# ---------------- TABS ----------------
tabs = st.tabs(["🏠 Home", "📊 Insights", "📈 Reports"])

# ---------------- HOME ----------------
with tabs[0]:
    st.markdown("""
    <h1 style='text-align:center; color:#4A2C2A; font-size:70px; font-family:"Cinzel Decorative", serif !important; font-weight:700;'>
    Sentiment Analyzer Dashboard
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analyze feedback with AI-powered insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='moving-text'><span>Decoding Feedback • Empowering Better Governance • Enhancing Customer Experience •</span></div>", unsafe_allow_html=True)


    st.markdown("""
    <div class="carousel-container">
        <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f" class="carousel-image" />
        <img src="https://images.unsplash.com/photo-1526378724218-0f7b2b1f4b6a" class="carousel-image" />
        <img src="https://images.unsplash.com/photo-1504384308090-c894fdcc538d" class="carousel-image" />
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='info-card'><h4>🧠 What is it?</h4><p>An AI-powered system that detects emotions in feedback.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='info-card'><h4>📊 How does it work?</h4><p>Type or speak comments; AI classifies their sentiment automatically.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='info-card'><h4>🎯 Why use it?</h4><p>Identify customer moods, improve services, and make data-driven decisions.</p></div>", unsafe_allow_html=True)


# ---------------- INSIGHTS ----------------
with tabs[1]:
    st.markdown("<h2 style='text-align:center; color:#4A2C2A; font-size:70px;'>SENTIMENT INSIGHTS</h2>", unsafe_allow_html=True)

    conn = create_connection()
    if conn:
        df_all = pd.read_sql("SELECT * FROM consultations ORDER BY timestamp DESC", conn)
        conn.close()
    else:
        df_all = pd.DataFrame(columns=["id","serial_no","comment","sentiment","confidence","timestamp"])

    total = len(df_all)
    pos = len(df_all[df_all["sentiment"] == "Positive"])
    neg = len(df_all[df_all["sentiment"] == "Negative"])
    neu = len(df_all[df_all["sentiment"] == "Neutral"])
    sar = len(df_all[df_all["sentiment"] == "Sarcastic"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='info-card'><h4>Total</h4><h2 style='color:#4A2C2A;'>{total}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='info-card'><h4>Positive</h4><h2 style='color:{COLOR_MAP['Positive']}'>{pos}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='info-card'><h4>Negative</h4><h2 style='color:{COLOR_MAP['Negative']}'>{neg}</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='info-card'><h4>Neutral</h4><h2 style='color:{COLOR_MAP['Neutral']}'>{neu}</h2></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='info-card'><h4>Sarcastic</h4><h2 style='color:{COLOR_MAP['Sarcastic']}'>{sar}</h2></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Add / Analyze a comment")

    col_serial, col_category = st.columns([1, 1.2])
    with col_serial:
        serial_no = st.text_input("Serial Number")
    with col_category:
        category = st.selectbox("Category", ["Cricket", "Ministry of Corporate Affairs", "Movies", "Politics"], index=None, placeholder="Select Category")

    # 🗣️ Comment input box (linked with session_state)
    comment_box = st.text_input(
        "Enter Comment (Press Enter to trigger camera):",
        key="comment_box",
        value=st.session_state.get("comment_text", "")
    )

    if st.button("📷 Start Camera for Face Recognition"):
        run_face_recognition()

        
    if st.button("🎙️ Speak Now"):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("🎧 Listening... Speak now")
                recognizer.adjust_for_ambient_noise(source, duration=0.7)
                audio = recognizer.listen(source, phrase_time_limit=6)
                text_from_speech = recognizer.recognize_google(audio, language="en-IN")
                st.session_state["comment_text"] = text_from_speech
                st.success("✅ Recognized and filled comment box.")
                st.rerun()
        except Exception as e:
            st.error(f"Speech recognition error: {e}")

    # Analyze
    uploaded_file = st.file_uploader("📂 Upload CSV file (with 'comment' column)", type=["csv"])
    if uploaded_file is not None:
        try:
            df_csv = pd.read_csv(uploaded_file)
            if 'comment' not in df_csv.columns:
                st.error("CSV must contain a 'comment' column.")
            else:
                with st.spinner("Analyzing uploaded data..."):
                    for _, row in df_csv.iterrows():
                        txt = str(row["comment"])
                        serial = str(row.get("serial_no", ""))
                        if model and vectorizer:
                            X = vectorizer.transform([txt])
                            if hasattr(model, "predict_proba"):
                               probs = model.predict_proba(X)[0]
                               idx = int(np.argmax(probs))
                               pred_label = model.classes_[idx]
                               pred_label = fix_sentiment(txt, pred_label)
                               conf = float(probs[idx]) * 100
                            

# 🧠 Hybrid fallback for low-confidence results
                            if conf < 60:
                                blob = TextBlob(txt)
                                tb_polarity = blob.sentiment.polarity
                                if tb_polarity > 0.1:
                                    pred_label = "Positive"
                                elif tb_polarity < -0.1:
                                    pred_label = "Negative"
                                else:
                                    pred_label = "Neutral"
                                conf = abs(tb_polarity) * 100

                            # 🚨 Mark irrelevant if meaningless
                            if not txt.strip() or conf < 20:
                                pred_label = "Irrelevant"


                                # 🧠 Handle unexpected output
                            if pred_label not in ["Positive", "Negative", "Neutral", "Sarcastic"]:
                                pred_label = "Irrelevant"

                            else:
                                pred_label = model.predict(X)[0]
                                conf = 100.0
                            conn = create_connection()
                            if conn:
                                cur = conn.cursor()
                                cur.execute("INSERT INTO consultations (serial_no, comment, sentiment, confidence) VALUES (%s,%s,%s,%s)",
                                            (serial, txt, pred_label, conf))
                                conn.commit()
                                cur.close()
                                conn.close()
                st.success("✅ CSV data uploaded and analyzed successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {e}")


    if st.button("Analyze Sentiment"):
        if not comment_box.strip():
            st.warning("Enter a comment first.")
        else:
            if model and vectorizer:
                X = vectorizer.transform([comment_box])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    idx = int(np.argmax(probs))
                    pred_label = model.classes_[idx]
                    conf = float(probs[idx]) * 100
                else:
                    pred_label = model.predict(X)[0]
                    conf = 100.0

# 🧠 Hybrid logic: fallback to TextBlob if confidence is low
                if conf < 60:
                    blob = TextBlob(comment_box)
                    tb_polarity = blob.sentiment.polarity  # range [-1, 1]

                    if tb_polarity > 0.1:
                        pred_label = "Positive"
                    elif tb_polarity < -0.1:
                        pred_label = "Negative"
                    else:
                        pred_label = "Neutral"
                    conf = abs(tb_polarity) * 100

                # 🚨 Final check for irrelevant text
                if not comment_box.strip() or (conf < 20):
                    pred_label = "Irrelevant"

                insert_record(serial_no, category, comment_box, pred_label, conf)
                st.success(f"Sentiment: {pred_label} | Confidence: {conf:.2f}%")


    if st.button("♻️ Reset All Data"):
        conn = create_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM consultations")
            conn.commit()
            cur.close()
            conn.close()
            st.warning("All records deleted!")
            st.rerun()

    if not df_all.empty:
        st.divider()
        st.subheader("📈 Visual Insights")

        col1, col2 = st.columns(2)
        with col1:
            st.write("☁️ Word Cloud")
            text_all = " ".join(df_all["comment"].astype(str))
            text_all = " ".join(df_all["comment"].dropna().astype(str))

# Clean the text
            text_all = re.sub(r'[^A-Za-z\s]', ' ', text_all)  # keep only letters and spaces
            text_all = re.sub(r'\s+', ' ', text_all).strip()

            if len(text_all.split()) > 2:
                wc = WordCloud(
                    width=800, 
                    height=400, 
                    background_color="white", 
                    colormap='viridis', 
                    collocations=False  # prevents joining "good service" as one token
                ).generate(text_all)

                fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Not enough valid words to generate a word cloud.")

                
        with col2:
            st.write("📊 Average Confidence by Sentiment")
            avg_conf = df_all.groupby("sentiment")["confidence"].mean().reset_index()
            ordered = ["Positive", "Neutral", "Negative", "Sarcastic"]
            avg_conf["sentiment"] = pd.Categorical(avg_conf["sentiment"], categories=ordered, ordered=True)
            avg_conf = avg_conf.sort_values("sentiment")
            fig_bar = px.bar(avg_conf, x="sentiment", y="confidence", color="sentiment", color_discrete_map=COLOR_MAP)
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("<h4 style='text-align:center; color:#4A2C2A;'>Sentiment Summary</h4>", unsafe_allow_html=True)
        colg1, colg2 = st.columns(2)

        st.markdown("### 📊 Sentiment Summary")

        colg1, colg2 = st.columns([1, 1])  # side-by-side layout

# --------- GAUGE METER (Left) ----------
        with colg1:
            st.write("🧭 Gauge Meter (Latest Record)")
            try:
                if "sentiment_label" in st.session_state and st.session_state["sentiment_label"] is not None:
                    latest_sent = st.session_state["sentiment_label"]
                    latest_conf = st.session_state.get("confidence", 0.0)
                elif not df_all.empty:
                    latest = df_all.iloc[0]
                    latest_sent = str(latest["sentiment"])
                    latest_conf = float(latest["confidence"]) if not pd.isna(latest["confidence"]) else 0.0
                else:
                    latest_sent = "No Data"
                    latest_conf = 0.0

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=latest_conf,
                    title={'text': latest_sent},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': COLOR_MAP.get(latest_sent, "#4A2C2A")}
                    }
                ))
                st.plotly_chart(fig_g, use_container_width=True)
            except Exception as e:
                st.info(f"No data for gauge meter. ({e})")

        # --------- PIE CHART (Right) ----------
        with colg2:
            st.write("🥧 Sentiment Distribution (Pie Chart)")
            if not df_all.empty:
                pie_data = df_all["sentiment"].value_counts().reset_index()
                pie_data.columns = ["sentiment", "count"]
                fig_pie = px.pie(
                    pie_data,
                    names="sentiment",
                    values="count",
                    color="sentiment",
                    color_discrete_map=COLOR_MAP,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data available for pie chart.")

        st.subheader("🔺 3D Sentiment Visualization")
        df_all["date"] = pd.to_datetime(df_all["timestamp"]).dt.date
        timeline = df_all.groupby(["date", "sentiment"]).size().reset_index(name="count")
        sent_order = {"Positive": 0, "Neutral": 1, "Negative": 2, "Sarcastic": 3}
        timeline["sent_num"] = timeline["sentiment"].map(sent_order)
        timeline["date_ordinal"] = pd.to_datetime(timeline["date"]).map(lambda d: d.toordinal())
        fig3d = go.Figure(data=[go.Scatter3d(
            x=timeline["date_ordinal"],
            y=timeline["sent_num"],
            z=timeline["count"],
            mode='markers+lines',
            marker=dict(size=6, color=timeline["sent_num"], colorscale='Viridis'),
            line=dict(width=2)
        )])
        fig3d.update_layout(scene=dict(
            xaxis_title='Date (ordinal)',
            yaxis_title='Sentiment (0=P,1=N,2=Neg,3=Sar)',
            zaxis_title='Count'
        ), height=600)
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("No records yet. Add or upload comments to see insights.")


# ---------------- REPORTS ----------------
with tabs[2]:
    st.markdown("<h2 style='text-align:center; color:#4A2C2A;'>Reports</h2>", unsafe_allow_html=True)
    conn = create_connection()
    if conn:
        df = pd.read_sql("SELECT * FROM consultations ORDER BY timestamp DESC", conn)
        conn.close()
    else:
        df = pd.DataFrame()

    if not df.empty:
        f1, f2 = st.columns([2,1])
        with f1:
            serial_search = st.text_input("🔍 Serial Number Contains:")
        with f2:
            sentiment_filter = st.selectbox("Filter Sentiment:", ["All"] + sorted(df["sentiment"].unique().tolist()))

        df_display = df.copy()
        if serial_search:
            df_display = df_display[df_display["serial_no"].astype(str).str.contains(serial_search, case=False, na=False)]
        if sentiment_filter != "All":
            df_display = df_display[df_display["sentiment"] == sentiment_filter]
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No records yet.")

    st.markdown("<hr style='border:1px solid #e0c4a3;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#4A2C2A;'>© 2025 Sentiment Analyzer | Designed with ❤️</p>", unsafe_allow_html=True)   