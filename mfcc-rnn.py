import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
import gradio as gr  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Toutes les bibliothèques ont été chargées avec succès !")

# === 1. Prétraitement : Nettoyage Audio ===
def preprocess_audio(file_path):
    """
    Nettoyer l'audio et extraire les caractéristiques Mel Spectrogram
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # Charger l'audio avec un taux d'échantillonnage fixe

        # Vérifie la durée minimale
        if len(audio) < sr * 0.2:  # Moins de 0.2 secondes
            print(f"Erreur : Le fichier audio {file_path} est trop court ({len(audio) / sr:.2f} sec). Ignoré.")
            return None

        # Réduction de bruit
        noise_profile = np.mean(audio)  # Estimation du profil de bruit
        audio = audio - noise_profile  # Réduction du bruit

        # Suppression des silences
        intervals = librosa.effects.split(audio, top_db=20)
        audio = np.concatenate([audio[start:end] for start, end in intervals])

        # Calcul du Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalisation et redimensionnement
        scaler = StandardScaler()
        log_mel_spectrogram = scaler.fit_transform(log_mel_spectrogram)
        log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=32, axis=1)

        print(f"Fichier traité avec succès : {file_path}")
        return log_mel_spectrogram.T  # Format : (32, 128)
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path} : {e}")
        return None


# === 2. Modèle RNN (LSTM) ===
def build_model(input_shape):
    """
    Construire un modèle RNN (basé sur LSTM) pour la classification de chiffres audio
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),  # Aplatir pour déterminer dynamiquement la forme
        layers.Dense(128, activation='relu'),
        layers.Reshape((8, -1)),  # Calcul dynamique de la deuxième dimension
        layers.LSTM(128, return_sequences=False),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Réduction du surapprentissage
        layers.Dense(11, activation='softmax')  # 10 chiffres + 1 classe "Inconnu"
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# === 3. Charger les données ===
def load_data(dataset_path):
    """
    Charger les fichiers audio et les étiquettes
    """
    X, y = [], []
    for label in range(10):  # Parcourt les classes de 0 à 9
        folder = os.path.join(dataset_path, str(label))
        
        # Vérifie si le dossier existe
        if not os.path.exists(folder):
            print(f"Avertissement : Le dossier {folder} est introuvable. Ignoré.")
            continue

        # Parcourt les fichiers dans le dossier
        for file in os.listdir(folder):
            if not file.lower().endswith(('.wav', '.mp3')):
                print(f"Ignorer le fichier non audio : {file}")
                continue

            file_path = os.path.join(folder, file)
            spectrogram = preprocess_audio(file_path)
            if spectrogram is not None:
                X.append(spectrogram)
                y.append(label)
    if not X or not y:
        raise ValueError("Aucune donnée valide trouvée dans le chemin du dataset spécifié.")
    X = np.array(X)[..., np.newaxis]  # Ajouter une dimension de canal
    y = np.array(y)
    print(f"{len(X)} échantillons ont été chargés depuis le dataset.")
    return X, y


# === Chemin du dataset et préparation ===
dataset_path = r"Data Scource"  # Remplacez par le chemin de votre dataset
X, y = load_data(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 4. Entraîner le modèle ===
model = build_model(input_shape=(32, 128, 1))
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Sauvegarder le modèle
model.save("digit_recognition_rnn.keras")

# === 5. Interface Gradio ===
def predict_digit(file_path):
    """
    Prédire le chiffre à partir d'un fichier audio avec un seuil de confiance
    """
    try:
        spectrogram = preprocess_audio(file_path)
        if spectrogram is not None:
            spectrogram = np.expand_dims(spectrogram, axis=0)[..., np.newaxis]
            prediction = model.predict(spectrogram)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)

            # Vérifie si la confiance est suffisante
            if confidence < 0.7:  # Seuil ajustable
                return "Erreur : Le modèle n'est pas suffisamment confiant dans sa prédiction. Réessayez."
            
            # Vérifie si la prédiction est un chiffre valide
            if 0 <= predicted_digit <= 9:
                return f"Le chiffre prédit est : {predicted_digit} (Confiance : {confidence:.2f})"
            elif predicted_digit == 10:
                return "Erreur : L'entrée n'est pas un chiffre valide (0-9)."
        else:
            return "Erreur : Impossible de traiter le fichier audio. Réessayez."
    except Exception as e:
        return f"Une erreur est survenue pendant la prédiction : {str(e)}"

interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Reconnaissance vocale des chiffres",
    description="Dites un chiffre (0-9) et le système le prédira."
)
interface.launch()
