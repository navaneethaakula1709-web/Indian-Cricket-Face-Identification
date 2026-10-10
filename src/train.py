import os
import pickle
import numpy as np

from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# ==============================
# Paths
# ==============================

DATASET_PATH = "dataset_clean"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)


# ==============================
# Store embeddings and labels
# ==============================

embeddings = []
labels = []


# ==============================
# Read images from dataset
# ==============================

print("\n================================")
print("   INDIAN CRICKETER FACE TRAINING")
print("================================")

EXCLUDE_FOLDERS = {
    "images",
    "media",
    "sample_data",
    "shared",
    "static",
    "tests",
    "test_images",
}

players = [
    player
    for player in sorted(os.listdir(DATASET_PATH))
    if os.path.isdir(os.path.join(DATASET_PATH, player))
    and player.lower() not in EXCLUDE_FOLDERS
]

print("Total genuine player folders:", len(players))
for player in sorted(players):

    player_folder = os.path.join(
        DATASET_PATH,
        player
    )

    print("\n==============================")
    print("Processing:", player)
    print("==============================")

    image_files = [
        image_name
        for image_name in os.listdir(player_folder)
        if image_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        )
    ]

    print("Images found:", len(image_files))

    for image_name in image_files:

        image_path = os.path.join(
            player_folder,
            image_name
        )

        try:

            result = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=True
            )

            embedding = result[0]["embedding"]

            embeddings.append(embedding)
            labels.append(player)

            print("✓", image_name)

        except Exception as e:

            print("✗ Skipped:", image_name)
            print("  Reason:", str(e))


# ==============================
# Check dataset
# ==============================

print("\n================================")
print("Embedding generation completed")
print("================================")

print("Total embeddings:", len(embeddings))
print("Total labels:", len(labels))


if len(embeddings) == 0:

    print("\nERROR: No faces were detected.")
    print("Please check your dataset images.")

    exit()


# ==============================
# Check number of classes
# ==============================

unique_players = sorted(set(labels))

print("\nPlayers successfully processed:")
print("Total classes:", len(unique_players))

for player in unique_players:
    print("-", player)


if len(unique_players) < 2:

    print("\nERROR: At least 2 players are required for training.")

    exit()


# ==============================
# Convert labels
# ==============================

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)


# ==============================
# Convert embeddings to numpy
# ==============================

X = np.array(embeddings)
y = np.array(encoded_labels)


print("\n================================")
print("Training data")
print("================================")

print("Embedding shape:", X.shape)
print("Number of samples:", len(X))
print("Number of classes:", len(label_encoder.classes_))


# ==============================
# Train SVM classifier
# ==============================

print("\nTraining SVM classifier...")

classifier = SVC(
    kernel="linear",
    probability=True
)

classifier.fit(
    X,
    y
)

print("\nTraining completed successfully!")


# ==============================
# Save model
# ==============================

model_data = {
    "classifier": classifier,
    "label_encoder": label_encoder
}


model_file = os.path.join(
    MODEL_PATH,
    "face_classifier.pkl"
)


with open(model_file, "wb") as f:

    pickle.dump(
        model_data,
        f
    )


# ==============================
# Final output
# ==============================

print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")

print("Model:", model_file)

print("\nClasses:")

for class_name in label_encoder.classes_:

    print("-", class_name)

print("\n================================")
print("TRAINING FINISHED")
print("================================")