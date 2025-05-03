#!/usr/bin/env python3
import os
import shutil
import subprocess
import time
import csv

# === Konfiguration ===
GOOD_FOLDER = "/Users/matthias/Documents/GitHub/Eagle_AI/GLASS/datasets/mvtec/thermal_image/test/good"
INPUT_FOLDER = "/Users/matthias/Documents/GitHub/Eagle_AI/GLASS/code_frontend/static/input"
SCRIPT_PATH = "/Users/matthias/Documents/GitHub/Eagle_AI/GLASS/shell/run-mvtec.sh"
ANNOTATED_FOLDER = "/Users/matthias/Documents/GitHub/Eagle_AI/GLASS/results/models/backbone_0/mvtec_thermal_image/annotated_scores"
CSV_PATH = "/Users/matthias/Documents/GitHub/Eagle_AI/GLASS/results/models/backbone_0/mvtec_thermal_image/image_scores.csv"
DEST_FOLDER = "/Users/matthias/Documents/GitHub/Eagle_AI/GLASS/code_frontend/static/images"
POLL_INTERVAL = 10  # Sekunden

# === Hilfsfunktionen ===

def clear_folder(path):
    """Löscht alle Dateien (keine Unterordner) in `path`."""
    for fname in os.listdir(path):
        fp = os.path.join(path, fname)
        if os.path.isfile(fp):
            os.remove(fp)

def copy_folder(src, dst):
    """Kopiert alle Dateien von `src` nach `dst`."""
    for fname in os.listdir(src):
        sp = os.path.join(src, fname)
        dp = os.path.join(dst, fname)
        if os.path.isfile(sp):
            shutil.copy2(sp, dp)

def read_scores(csv_file):
    """
    Liest image_scores.csv und gibt dict zurück:
    { "filename.jpg": score_float, ... }
    """
    scores = {}
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Annahme: Spalten heißen 'filename' und 'score'
            fname = row.get('Image Name') or row.get('file') or row.get('image')
            try:
                score = float(row['Score'])
            except:
                continue
            scores[fname] = score
    return scores

# === Main-Pipeline ===

def main():
    # 1) GOOD_FOLDER leeren
    os.makedirs(GOOD_FOLDER, exist_ok=True)
    print("1) Leere GOOD_FOLDER...")
    clear_folder(GOOD_FOLDER)

    # 2) INPUT_FOLDER → GOOD_FOLDER kopieren
    print("2) Kopiere Input-Bilder → GOOD_FOLDER...")
    copy_folder(INPUT_FOLDER, GOOD_FOLDER)

    # 3) INPUT_FOLDER leeren
    print("3) Leere INPUT_FOLDER...")
    clear_folder(INPUT_FOLDER)

    # 4) Bash-Skript ausführen
    print(f"4) Starte MVTec-Skript: {SCRIPT_PATH}")
    try:
        subprocess.run(
            ['bash', SCRIPT_PATH],
            cwd=os.path.dirname(SCRIPT_PATH),
            check=True
        )
        print("   → Skript erfolgreich beendet.")
    except subprocess.CalledProcessError as e:
        print("   → Fehler beim Ausführen des Skripts:", e)
        return

    # 5) Polling: auf annotated_scores warten
    print("5) Überwache Annotated-Scores (interval: {}s)...".format(POLL_INTERVAL))
    seen = set()
    while True:
        # 5a) CSV neu einlesen
        scores = read_scores(CSV_PATH)
        # 5b) alle Dateien im Annotated-Ordner
        for fname in os.listdir(ANNOTATED_FOLDER):
            if fname in seen:
                continue
            seen.add(fname)
            full_path = os.path.join(ANNOTATED_FOLDER, fname)
            score = scores.get(fname, None)
            if score is not None and score > 0.98:
                # 5c) kopieren
                dst = os.path.join(DEST_FOLDER, fname)
                print(f"    → Kopiere {fname} (score={score}) → {DEST_FOLDER}")
                shutil.copy2(full_path, dst)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
