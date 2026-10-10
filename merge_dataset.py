
import os
import shutil

SOURCE = r"C:\Users\navan\Downloads\archive\indian cricketer"
DEST = "dataset_clean"

players = [
    "Shubman Gill",
    "Rishabh Pant",
    "Suryakumar Yadav",
    "Sanju Samson",
    "Shreyas Iyer",
    "Ishan Kishan",
    "Mohammed Siraj",
    "Ruturaj Gaikwad",
    "Yashasvi Jaiswal",
    "Smriti Mandhana",
    "Harmanpreet Kaur",
    "Jemimah Rodrigues",
    "Deepti Sharma",
    "Shafali Verma",
    "Richa Ghosh",
    "Harleen Deol",
    "Radha Yadav",
    "Shubhlakshmi Sharma",
    "Sneh Rana",
    "Nitish Rana",
]

extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

folders = {
    name.casefold(): name
    for name in os.listdir(SOURCE)
    if os.path.isdir(os.path.join(SOURCE, name))
}

for player in players:
    source_name = folders.get(player.casefold())

    if not source_name:
        print("NOT FOUND:", player)
        continue

    source_folder = os.path.join(SOURCE, source_name)
    target_name = player.replace(" ", "_")
    target_folder = os.path.join(DEST, target_name)

    if os.path.exists(target_folder):
        print("SKIPPED existing:", player)
        continue

    images = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(extensions):
                images.append((os.path.join(root, file), file))

    if not images:
        print("NO IMAGES:", player)
        continue

    os.makedirs(target_folder)
    for src, filename in images:
        shutil.copy2(src, os.path.join(target_folder, filename))

    print("ADDED:", player, "-", len(images), "images")

print("Finished. Check the output above.")
