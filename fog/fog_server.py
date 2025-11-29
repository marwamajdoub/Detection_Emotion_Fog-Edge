import socket
import pickle
import struct
import cv2
import numpy as np
from fer import FER  # Assurez-vous d'avoir installé `pip install fer`

# Initialisation du modèle FER
detector = FER(mtcnn=True)

def predict_emotion(img: np.ndarray):
    """
    Prédit les émotions pour une image donnée.
    Retourne une liste de dictionnaires avec les émotions et les scores de confiance.
    """
    results = detector.detect_emotions(img)
    if not results:
        return []

    faces_emotions = []
    for face in results:
        emotions = face["emotions"]
        main_emotion = max(emotions, key=emotions.get)
        confidence = emotions[main_emotion]
        faces_emotions.append({
            "emotion": main_emotion,
            "confidence": confidence
        })
    return faces_emotions

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
                face_img = pickle.loads(data)
                if face_img is None:
                    print("[ERROR] Image invalide reçue")
                    continue

                # Prédire les émotions
                faces_emotions = predict_emotion(face_img)
                print(f"[FOG] Détecté {len(faces_emotions)} visage(s) : {faces_emotions}")

                # Envoyer la réponse au Edge
                response = {
                    "status": "success",
                    "faces": faces_emotions,
                    "height": face_img.shape[0],
                    "width": face_img.shape[1]
                }
                response_data = pickle.dumps(response)
                conn.sendall(struct.pack(">L", len(response_data)))
                conn.sendall(response_data)
