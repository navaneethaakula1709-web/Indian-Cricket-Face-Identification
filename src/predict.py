import os
import sys
import json
import pickle
import numpy as np

from deepface import DeepFace

# ==============================
# Paths
# ==============================

DATA_PATH = "data/indian_cricketers.json"
MODEL_PATH = "models/face_classifier.pkl"

# ==============================
# Validate image path
# ==============================

if len(sys.argv) < 2:
    print("Please provide an image path.")
    print("Example: python src/predict.py test_images/Dhoni.webp")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("Image not found:", image_path)
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print("Trained model not found. Run: python src/train.py")
    sys.exit(1)

# ==============================
# Load trained model
# ==============================

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

classifier = model_data["classifier"]
label_encoder = model_data["label_encoder"]

# ==============================
# Load player details
# ==============================

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        player_details = json.load(f)
else:
    player_details = {}

# ==============================
# Analyze image
# ==============================

print("\nAnalyzing image...")
print("Image:", image_path)

try:
    result = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        detector_backend="retinaface",
        enforce_detection=True
    )

    embedding = np.array(
        result[0]["embedding"]
    ).reshape(1, -1)

except Exception as e:
    print("\nError while processing image:")
    print(e)
    sys.exit(1)

# ==============================
# Predict player
# ==============================

prediction = classifier.predict(embedding)
player_name = label_encoder.inverse_transform(prediction)[0]

probabilities = classifier.predict_proba(embedding)[0]
confidence = float(np.max(probabilities) * 100)

# ==============================
# Display prediction
# ==============================

print("\n" + "=" * 45)
print("             PREDICTION")
print("=" * 45)

print("Player:", player_name)
print(f"Confidence: {confidence:.2f}%")

# ==============================
# Display player biography
# ==============================

details = player_details.get(player_name)

if details is None:
    # Try matching names if folder and JSON key formats differ.
    normalized_name = "".join(
        ch.lower() for ch in player_name if ch.isalnum()
    )

    for key, value in player_details.items():
        normalized_key = "".join(
            ch.lower() for ch in key if ch.isalnum()
        )

        if normalized_key == normalized_name:
            details = value
            break

if details:
    print("\n" + "=" * 45)
    print("           PLAYER BIO")
    print("=" * 45)

    print("Full name:", details.get("full_name", player_name))
    print("Country:", details.get("country", "Not available"))
    print("Team:", details.get("team", "Not available"))
    print("Role:", details.get("role", "Not available"))
    print("Batting style:", details.get("batting_style", "Not available"))
    print("Bowling style:", details.get("bowling_style", "Not available"))
    print("Jersey number:", details.get("jersey_number", "Not available"))
    print("Date of birth:", details.get("date_of_birth", "Not available"))
    print("Career status:", details.get("career_status", "Not available"))

    print("\nBio:")
    print(details.get("bio", "Not available"))

    print("\nCareer history:")
    print(details.get("career_history", "Not available"))

    print("\nMajor achievements:")

    achievements = details.get("major_achievements", [])

    if achievements:
        for achievement in achievements:
            print("-", achievement)
    else:
        print("Not available")

else:
    print("\nPlayer bio not available in indian_cricketers.json.")
    print("Add a matching player entry to display biography.")

print("\n" + "=" * 45)