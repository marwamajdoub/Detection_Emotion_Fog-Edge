import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import plotly.express as px
import time

# -----------------------
# Initialiser Firebase
# -----------------------
cred = credentials.Certificate("../fog/serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------
# Titre du dashboard
# -----------------------
st.title("EmotionEdge Dashboard")
# -----------------------
# Sélection de la salle
# -----------------------
room_id = st.selectbox("Sélectionnez une salle", ["Salle1", "Salle2"])

# -----------------------
# Placeholder unique
# -----------------------
placeholder = st.empty()

# -----------------------
# Fonction pour récupérer toutes les émotions
# -----------------------
def get_all_emotions(room_id):
    docs = db.collection("rooms").document(room_id).collection("emotion_stats") \
             .order_by("timestamp", direction=firestore.Query.DESCENDING) \
             .stream()
    emotions = []
    for doc in docs:
        data = doc.to_dict()
        for face in data["faces"]:
            emotions.append({
                "timestamp": data["timestamp"],
                "emotion": face["emotion"],
                "confidence": face["confidence"]
            })
    return emotions

# -----------------------
# Fonction pour mettre à jour le dashboard
# -----------------------
def update_dashboard():
    emotions = get_all_emotions(room_id)

    with placeholder.container():
        if emotions:
            df = pd.DataFrame(emotions)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

            # Graphique des émotions avec clé unique
            st.subheader("Répartition des émotions")
            emotion_counts = df["emotion"].value_counts().reset_index()
            emotion_counts.columns = ["emotion", "count"]
            fig = px.pie(emotion_counts, names="emotion", values="count")
            st.plotly_chart(fig, key=f"emotion_chart_{int(time.time()*1000)}")

            # Tableau des émotions
            st.subheader("Toutes les émotions détectées")
            st.dataframe(df)
        else:
            st.warning("Aucune émotion détectée pour cette salle.")

# -----------------------
# Bouton manuel pour rafraîchir
# -----------------------
st.button("Rafraîchir le dashboard", on_click=update_dashboard)

# -----------------------
# Affichage initial
# -----------------------
update_dashboard()
