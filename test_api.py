import random
import time
import uuid
from pathlib import Path

import requests

# Konfiguration
API_URL = "http://127.0.0.1:8000/api/v1/inference/predict"
DATASET_DIR = Path(
    "/home/ubuntu/dev/RottenBot_ExpTracking/training_datasets/rottenbot_all_classesv1/test"
)


def load_test_api():
    # Alle Ordner im Dataset-Verzeichnis finden
    category_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]

    if not category_dirs:
        print(f"Keine Ordner in {DATASET_DIR} gefunden!")
        return

    print(f"Gefunden: {len(category_dirs)} Kategorien")
    print("\nStarte Load Test - Drücke Ctrl+C zum Stoppen\n")

    request_count = 0
    first_request = True

    try:
        while True:
            # Zufälligen Ordner auswählen
            category_dir = random.choice(category_dirs)

            # Alle Bilder in diesem Ordner finden (alle Dateien)
            image_files = [f for f in category_dir.iterdir() if f.is_file()]

            if not image_files:
                continue

            # Zufälliges Bild auswählen
            image_path = random.choice(image_files)

            # Zufällige Parameter
            save_prediction = random.choice([True, False])
            user_uid = str(uuid.uuid4())

            # Request vorbereiten
            params = {"save_prediction": save_prediction, "user_uid": user_uid}

            if first_request:
                print(
                    f"Sende ersten Request (Modell lädt, kann länger dauern)...",
                    end=" ",
                    flush=True,
                )
                timeout = 120  # 2 Minuten für ersten Request
            else:
                print(
                    f"Sende Request #{request_count + 1} | {category_dir.name}...",
                    end=" ",
                    flush=True,
                )
                timeout = 30

            try:
                with open(image_path, "rb") as img_file:
                    files = {"file": (image_path.name, img_file, "image/jpeg")}
                    response = requests.post(
                        API_URL, params=params, files=files, timeout=timeout
                    )

                    request_count += 1
                    first_request = False

                    if response.status_code == 200:
                        data = response.json()
                        print(
                            f"✓ {data.get('predicted_class_name', 'N/A')} | "
                            f"{data.get('confidence', 0):.2%}"
                        )
                    else:
                        print(f"✗ Status: {response.status_code}")

            except requests.exceptions.Timeout:
                print(f"✗ Timeout ({timeout}s)!")
                if first_request:
                    print("Server antwortet nicht. Prüfe ob er läuft:")
                    print("  uvicorn src.main:app --reload")
                    break
            except Exception as e:
                print(f"✗ Error: {e}")  # 50ms Pause

    except KeyboardInterrupt:
        print(f"\n\nLoad Test beendet. Total: {request_count}")


if __name__ == "__main__":
    load_test_api()
