import os
import json
import pickle
import tempfile

import numpy as np
import streamlit as st
from deepface import DeepFace


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Indian Cricket Face Identification",
    page_icon="🏏",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================
st.title("🏏 Indian Cricket Face Identification")
st.write(
    "Upload an Indian cricketer's image to identify "
    "the player and view their profile."
)


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model_path = os.path.join(
        "models",
        "face_classifier.pkl"
    )

    with open(model_path, "rb") as file:
        model_data = pickle.load(file)

    return (
        model_data["classifier"],
        model_data["label_encoder"]
    )


# ==========================================
# LOAD PLAYER DATABASE
# ==========================================
@st.cache_data
def load_players():
    database_path = os.path.join(
        "data",
        "indian_cricketers.json"
    )

    with open(database_path, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    classifier, label_encoder = load_model()
    players = load_players()

except Exception as error:
    st.error("Unable to load the model or player database.")
    st.code(str(error))
    st.stop()


# ==========================================
# IMAGE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "📷 Upload Cricketer Image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==========================================
# HELPER FUNCTION
# ==========================================
def display_value(value):
    """Display a useful value instead of None or an empty string."""
    if value is None or value == "":
        return "Not available"
    return value


def find_player(player_key):
    """Find a player using the exact model label first."""
    player = players.get(player_key)

    if player is not None:
        return player

    # Try matching normalized keys if exact key is not found.
    normalized_key = player_key.strip().lower().replace(" ", "_")

    for key, value in players.items():
        if key.strip().lower().replace(" ", "_") == normalized_key:
            return value

    return None


# ==========================================
# IDENTIFICATION
# ==========================================
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Cricketer Image",
        width=350
    )

    identify_button = st.button(
        "🔍 Identify Cricketer",
        type="primary"
    )

    if identify_button:

        temp_path = None

        try:
            with st.spinner(
                "Analyzing image and identifying the cricketer..."
            ):

                # Save uploaded image temporarily
                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name

                # Generate face embedding
                result = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    detector_backend="retinaface",
                    enforce_detection=True
                )

                if not result:
                    st.error("No face detected in the uploaded image.")
                    st.stop()

                embedding = np.array(
                    result[0]["embedding"],
                    dtype=np.float32
                ).reshape(1, -1)

                # Predict player
                prediction = classifier.predict(embedding)

                player_key = str(
                    label_encoder.inverse_transform(prediction)[0]
                )

                # Calculate model confidence when supported
                confidence = None

                if hasattr(classifier, "predict_proba"):
                    probabilities = classifier.predict_proba(
                        embedding
                    )[0]

                    confidence = float(
                        np.max(probabilities) * 100
                    )

                # Find player profile
                player = find_player(player_key)

            # ==========================================
            # DISPLAY PLAYER RESULT
            # ==========================================
            if player is None:

                st.warning(
                    "The model predicted a player, but the matching "
                    "profile was not found in the JSON database."
                )

                st.write("Predicted label:", player_key)

            else:

                st.success("✅ Cricketer Identified!")

                st.header(
                    "🏏 " + str(
                        player.get("full_name", player_key)
                    )
                )

                if confidence is not None:
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                # ==========================================
                # PLAYER PROFILE
                # ==========================================
                st.divider()
                st.subheader("👤 Player Profile")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        "**Full Name:**",
                        display_value(
                            player.get("full_name", player_key)
                        )
                    )
                    st.write(
                        "**Country:**",
                        display_value(player.get("country"))
                    )
                    st.write(
                        "**Team:**",
                        display_value(player.get("team"))
                    )
                    st.write(
                        "**Role:**",
                        display_value(player.get("role"))
                    )
                    st.write(
                        "**Jersey Number:**",
                        display_value(player.get("jersey_number"))
                    )

                with col2:
                    st.write(
                        "**Date of Birth:**",
                        display_value(player.get("date_of_birth"))
                    )
                    st.write(
                        "**Batting Style:**",
                        display_value(player.get("batting_style"))
                    )
                    st.write(
                        "**Bowling Style:**",
                        display_value(player.get("bowling_style"))
                    )
                    st.write(
                        "**Career Status:**",
                        display_value(player.get("career_status"))
                    )

                # ==========================================
                # BIOGRAPHY
                # ==========================================
                st.divider()
                st.subheader("📖 Biography")

                biography = player.get(
                    "bio",
                    "Biography not available."
                )

                st.write(display_value(biography))

                # ==========================================
                # CAREER HISTORY
                # ==========================================
                st.divider()
                st.subheader("🏏 Cricket History")

                career_history = player.get(
                    "career_history",
                    "Career history not available."
                )

                st.write(display_value(career_history))

                # ==========================================
                # ACHIEVEMENTS
                # ==========================================
                st.divider()
                st.subheader("🏆 Major Achievements")

                achievements = player.get(
                    "major_achievements",
                    []
                )

                if isinstance(achievements, list) and achievements:
                    for achievement in achievements:
                        st.markdown(f"- {achievement}")

                elif isinstance(achievements, str) and achievements:
                    st.write(achievements)

                else:
                    st.info(
                        "Major achievements are not available "
                        "in the player database."
                    )

        except Exception as error:
            st.error(
                "❌ Could not identify the cricketer."
            )
            st.code(str(error))

        finally:
            # Remove temporary image file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
