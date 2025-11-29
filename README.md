# EmotionEdge: Système de Détection d’Émotions en Temps Réel

## 1. Description du Projet
EmotionEdge est un système de simulation de détection d’émotions en temps réel utilisant une architecture **Edge** + **Fog** + **Cloud** (Firebase), sans matériel physique.
Le système permet de visualiser sur un **dashboard les émotions détectées par des caméras simulées**.

## 2. Objectifs :
- Simuler un flux vidéo et détecter les visages -> **Noeud Edge**.
- Analyser les émotions avec un modèle IA léger -> **Noeud Fog**.
- Stocker les statistiques dans Firebase pour une récupération en temps réel.
- Afficher les résultats sur un dashboard interactif -> **Streamlit**.

## 2. Architecture du systéme 

### Flux de données :

-**Edge (PC 1)** : Simule des caméras, détecte et prétraite les visages.

-**Fog (PC 2)** : Analyse les émotions à l’aide d’un modèle pré-entraîné, agrège les statistiques et envoie les résultats à Firebase.

-**Firebase (Cloud)** : Stocke les statistiques et permet au dashboard de récupérer les données instantanément.

-**Dashboard (PC 3)** : Streamlit se connecte à Firebase pour afficher les statistiques en temps réel.

### Communication :

-**Edge → Fog** : via **Sockets TCP/IP** .

-**Fog → Firebase** : via Firebase Admin SDK (Python).

-**Firebase → Dashboard** : récupération en temps réel avec Streamlit.

## 3.Utilisation


### 1.Edge

Simule la caméra et détecte les visages.
```bash
python edge_camera.py
```

Les images sont envoyées au Fog via sockets.


### 2. Fog
Reçoit les images du Edge, analyse les émotions et envoie les résultats à Firebase.
```bash
python fog_server.py
```
Utilise un modèle pré-entraîné


### 3. Dashboard

Visualise les statistiques depuis Firebase.
```bash
streamlit run dashboard.py
```

Affiche les graphiques et tableaux des émotions détectées.
```
- **dashboard**: `http://localhost:8501/`
