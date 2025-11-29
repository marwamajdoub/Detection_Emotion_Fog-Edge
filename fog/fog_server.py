import socket
import pickle
import struct
import cv2
import numpy as np
from fer import FER  # Version 22.5.1
import traceback
import time
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Désactiver les optimisations OneDNN de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialiser Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialisation du détecteur FER (sans arguments supplémentaires)
detector = FER()  # Aucun argument n'est nécessaire pour cette version

def predict_emotion(img: np.ndarray):
    """
    Prédit les émotions dans une image avec des vérifications améliorées.
    Retourne une liste de dictionnaires avec les émotions et les scores de confiance.
    """
    try:
        # Vérification de l'image
        if img is None or img.size == 0:
            print("[WARN] Image vide ou invalide")
            return []

        # Conversion BGR → RGB (obligatoire pour FER)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Détection des émotions
        results = detector.detect_emotions(img_rgb)

        if not results:
            print("[WARN] Aucun visage détecté")
            return []

        faces_emotions = []
        for face in results:
            # Vérification de la taille du visage
            (x, y, w, h) = face["box"]
            if w < 50 or h < 50:  # Ignore les visages trop petits
                print(f"[WARN] Visage trop petit ({w}x{h}) - ignoré")
                continue

            # Extraction des émotions
            emotions = face["emotions"]
            main_emotion = max(emotions, key=emotions.get)
            confidence = emotions[main_emotion]

            # Filtre les détections peu confiantes
            if confidence > 0.3:  # Seuil minimal de confiance
                faces_emotions.append({
                    "emotion": main_emotion,
                    "confidence": float(confidence)
                })
                print(f"[DEBUG] Émotion détectée: {main_emotion} ({confidence:.2f})")

        return faces_emotions

    except Exception as e:
        print(f"[ERROR] Erreur dans predict_emotion: {str(e)}")
        traceback.print_exc()
        return []

def save_emotion_to_firestore(cam_id, faces_emotions, timestamp):
    """
    Sauvegarde les émotions détectées dans Firestore.
    """
    try:
        doc_ref = db.collection("rooms").document(cam_id).collection("emotion_stats").document()
        doc_ref.set({
            "timestamp": timestamp,
            "faces": faces_emotions,
        })
        print(f"[FIREBASE] Statistiques sauvegardées pour {cam_id}")
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'enregistrement dans Firestore: {e}")
        traceback.print_exc()

# ----------------------------------------
# Serveur Fog
# ----------------------------------------
HOST = "0.0.0.0"  # Écoute sur toutes les interfaces
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"[INFO] Serveur Fog en écoute sur {HOST}:{PORT}")
    while True:
        try:
            conn, addr = s.accept()
            with conn:
                print(f"[INFO] Connecté à {addr}")
                while True:
                    # Recevoir la taille de l'image
                    data = b""
                    while len(data) < 4:
                        packet = conn.recv(4 - len(data))
                        if not packet:
                            break
                        data += packet
                    if not data:
                        break

                    size = struct.unpack(">L", data)[0]

                    # Recevoir l'image
                    data = b""
                    while len(data) < size:
                        packet = conn.recv(size - len(data))
                        if not packet:
                            break
                        data += packet
                    if not data:
                        break

                    # Désérialiser l'image
                    try:
                        face_img = pickle.loads(data)
                        if face_img is None or face_img.size == 0:
                            print("[ERROR] Image vide ou corrompue")
                            continue
                        print(f"[DEBUG] Image reçue - Shape: {face_img.shape}, Type: {face_img.dtype}")
                        cv2.imshow("Image reçue", face_img)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"[ERROR] Erreur lors de la désérialisation: {str(e)}")
                        continue

                    # Prédire les émotions
                    faces_emotions = predict_emotion(face_img)
                    print(f"[FOG] Détecté {len(faces_emotions)} visage(s): {faces_emotions}")

                    # Sauvegarder dans Firebase
                    timestamp = time.time()
                    save_emotion_to_firestore("Salle1", faces_emotions, timestamp)

                    # Envoyer la réponse au Edge
                    response = {
                        "status": "success" if faces_emotions else "no_faces",
                        "faces": faces_emotions,
                        "height": face_img.shape[0],
                        "width": face_img.shape[1]
                    }
                    response_data = pickle.dumps(response)
                    conn.sendall(struct.pack(">L", len(response_data)))
                    conn.sendall(response_data)

        except Exception as e:
            print(f"[ERROR] Erreur de connexion: {str(e)}")
            traceback.print_exc()
            continue
