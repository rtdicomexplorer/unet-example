
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


# 🦷 Karies-Segmentierung mit U-Net – Projektübersicht

## 1. Daten vorbereiten
Lade deinen Datensatz herunter, z. B. von Kaggle.

### Dataset

- https://www.kaggle.com/datasets/kvipularya/a-collection-of-dental-x-ray-images-for-analysis



Stelle sicher:

all-images/ enthält alle Zahn-Röntgenbilder (.png)

unet-masks/ enthält die passenden binären Masken (weiß = Karies, schwarz = Hintergrund)

###  2. Train/Val Split
Führe dieses Skript aus: split_unet_data.py

🔁 Es verteilt Bilder und Masken automatisch im Verhältnis 80/20:


- dataset/
- ├── images/
- │   ├── train/
- │   └── val/
- ├── masks/
- │   ├── train/
- │   └── val/

### 3. DataLoader erstellen
→ karies-segmentation-dataset

Lädt Bild-Maske-Paare, resized auf 512×256, normalisiert und gibt Tensors zurück.
Wird später vom Training & predict verwendet.

### 4. U-Net Architektur definieren
→ unet_model.py

Implementiert ein klassisches U-Net-Modell für binäre Segmentierung.
Ausgabe: Maske mit Werten zwischen 0–1 (per torch.sigmoid)

### 5. Modell trainieren
→ train_unet.py

Lädt Trainingsdaten

Optimiert mit BCELoss

Speichert Modell nach Training als weights/unet_karies.pt

Zeigt Loss-Kurve (Train/Val)

###  6. Inferenz auf neuen Bildern
→ predict.py

Lädt gespeichertes Modell

Macht Vorhersage auf Bild

Zeigt Overlay von Originalbild + Vorhersagemaske






