import cv2
import socket
import pickle
import struct
import time
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Edge camera simulator (video + face detection)")
    parser.add_argument("--source", type=str, default="0", help="Chemin vers la vidéo ou '0' pour webcam")
    parser.add_argument("--fog-ip", type=str, default="127.0.0.1", help="Adresse IP du serveur Fog")
    parser.add_argument("--fog-port", type=int, default=65432, help="Port du serveur Fog")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalle en secondes entre deux envois au Fog")
    return parser.parse_args()

def init_capture(source):
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source vidéo: {source}")
    return cap

def init_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Impossible de charger le classifieur Haar Cascade")
    return face_cascade

def detect_and_preprocess_faces(frame, face_cascade, face_size=(224, 224)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    face_images = [cv2.resize(frame[y:y+h, x:x+w], face_size) for (x, y, w, h) in faces]
    return face_images, faces

def send_to_fog(face_images, fog_ip, fog_port, cam_id="edge_cam_1"):
    if not face_images:
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((fog_ip, fog_port))
            print(f"[INFO] Connecté au serveur Fog ({fog_ip}:{fog_port})")

            for idx, face_img in enumerate(face_images):
                data = pickle.dumps(face_img)
                s.sendall(struct.pack(">L", len(data)))
                s.sendall(data)
                print(f"[INFO] Visage {idx} envoyé au Fog")

                # Recevoir la réponse du Fog
                response_data = b""
                while len(response_data) < 4:
                    response_data += s.recv(4 - len(response_data))
                response_size = struct.unpack(">L", response_data)[0]

                response_data = b""
                while len(response_data) < response_size:
                    packet = s.recv(response_size - len(response_data))
                    if not packet:
                        break
                    response_data += packet

                if response_data:
                    response = pickle.loads(response_data)
                    print(f"[INFO] Réponse du Fog : {response}")

    except socket.error as e:
        print(f"[ERROR] Erreur de connexion au Fog: {e}")
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'envoi: {e}")

def main():
    args = parse_args()
    cap = init_capture(args.source)
    face_cascade = init_face_detector()
    print("[INFO] Démarrage du simulateur Edge")
    last_send_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        face_images, faces = detect_and_preprocess_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Edge Camera - Preview", frame)

        if time.time() - last_send_time >= args.interval:
            send_to_fog(face_images, args.fog_ip, args.fog_port)
            last_send_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
