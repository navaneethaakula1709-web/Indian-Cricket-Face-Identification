import os
import shutil
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("dataset")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# FUNCTIONS
# ============================================================

def normalize_name(name):
    """
    Makes player names comparable.

    Example:
    MS Dhoni  -> msdhoni
    Ms_Dhoni  -> msdhoni
    """

    return "".join(
        char.lower()
        for char in name
        if char.isalnum()
    )


def get_existing_players():
    """
    Get existing player folders.
    """

    players = {}

    if not DATASET_PATH.exists():
        DATASET_PATH.mkdir(parents=True)

    for folder in DATASET_PATH.iterdir():

        if folder.is_dir():

            key = normalize_name(folder.name)

            players[key] = folder

    return players


def find_image_folders(source_path):
    """
    Find folders containing player images.
    """

    folders = []

    for root, dirs, files in os.walk(source_path):

        image_files = [
            f for f in files
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ]

        if image_files:
            folders.append(
                (
                    Path(root),
                    image_files
                )
            )

    return folders


# ============================================================
# MAIN
# ============================================================

print("\n==========================================")
print("   INDIAN CRICKETERS DATASET EXPANSION")
print("==========================================\n")

print("Existing dataset:")
print(DATASET_PATH.resolve())

existing_players = get_existing_players()

print("\nExisting players:")

for folder in existing_players.values():
    print(" -", folder.name)


# ============================================================
# ASK FOR SOURCE DATASET LOCATION
# ============================================================

print("\n------------------------------------------")
print("Enter the path of your downloaded Kaggle")
print("Indian Cricketers dataset folder.")
print("------------------------------------------")

source_input = input("\nDataset path: ").strip()

source_path = Path(source_input.strip('"'))

if not source_path.exists():

    print("\nERROR: Dataset path not found.")
    print("Please check the path and run again.")

    exit()


# ============================================================
# FIND IMAGE FOLDERS
# ============================================================

print("\nScanning dataset...")

image_folders = find_image_folders(source_path)

if not image_folders:

    print("\nERROR: No image folders found.")
    exit()


print("\nImage folders found:", len(image_folders))


# ============================================================
# COPY DATA
# ============================================================

total_copied = 0
total_skipped = 0

new_players = set()

for folder_path, image_files in image_folders:

    player_name = folder_path.name.strip()

    if not player_name:
        continue

    player_key = normalize_name(player_name)

    # Check whether player already exists
    if player_key in existing_players:

        destination = existing_players[player_key]

    else:

        # Convert spaces to underscores
        clean_name = player_name.replace(" ", "_")

        destination = DATASET_PATH / clean_name

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        existing_players[player_key] = destination

        new_players.add(destination.name)

        print("\nNEW PLAYER:", destination.name)

    # Copy images
    for image_name in image_files:

        source_file = folder_path / image_name

        destination_file = destination / image_name

        # Avoid overwriting existing images
        if destination_file.exists():

            total_skipped += 1
            continue

        try:

            shutil.copy2(
                source_file,
                destination_file
            )

            total_copied += 1

        except Exception as e:

            print(
                "Could not copy:",
                source_file
            )

            print("Reason:", e)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("          DATASET EXPANSION DONE")
print("==========================================")

print("\nNew players added:", len(new_players))

for player in sorted(new_players):

    print(" +", player)

print("\nImages copied:", total_copied)

print("Images skipped:", total_skipped)

print(
    "\nTotal player folders now:",
    len(existing_players)
)

print("\nDataset location:")
print(DATASET_PATH.resolve())

print("\n==========================================")
print("NEXT STEP")
print("==========================================")

print("\nRun:")
print("python src/train.py")

print("\nThis will retrain the model with")
print("all players currently inside dataset/.")