import os
import shutil
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

CSV_PATH = "players.csv"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "dataset"


# -----------------------------
# Players selected for project
# -----------------------------

SELECTED_PLAYERS = [
    "hardik_pandya",
    "jasprit_bumrah",
    "ms_dhoni",
    "rohit_sharma",
    "virat_kohli"
]


# -----------------------------
# Read CSV
# -----------------------------

df = pd.read_csv(CSV_PATH)

print("CSV loaded successfully!")
print("Total images in CSV:", len(df))


# -----------------------------
# Process selected players
# -----------------------------

for player in SELECTED_PLAYERS:

    player_data = df[df["player"] == player]

    print("\n-----------------------------")
    print("Processing:", player)
    print("Images found:", len(player_data))

    # Output folder name
    output_player_folder = os.path.join(
        OUTPUT_FOLDER,
        player.title().replace(" ", "_")
    )

    # If a file with same name exists, remove it
    if os.path.exists(output_player_folder) and not os.path.isdir(output_player_folder):
        os.remove(output_player_folder)

    # Create folder
    os.makedirs(output_player_folder, exist_ok=True)

    copied = 0

    # -----------------------------
    # Copy images
    # -----------------------------

    for image_name in player_data["image"]:

        # IMPORTANT:
        # Images are inside player-wise folders
        source_path = os.path.join(
            IMAGE_FOLDER,
            player,
            image_name
        )

        destination_path = os.path.join(
            output_player_folder,
            image_name
        )

        if os.path.exists(source_path):

            shutil.copy2(
                source_path,
                destination_path
            )

            copied += 1

        else:
            print("Image not found:", source_path)

    print("Images copied:", copied)


print("\n================================")
print("DATASET PREPARATION COMPLETED!")
print("================================")