import os
import shutil
from pathlib import Path
import random

# === Konfiguration ===
random.seed(42)
split_ratio = 0.8  # 80 % Training, 20 % Validierung

src_images = Path("all-images")
src_masks = Path("unet-masks")

dst_root = Path("dataset")
dst_img_train = dst_root / "images" / "train"
dst_img_val = dst_root / "images" / "val"
dst_mask_train = dst_root / "masks" / "train"
dst_mask_val = dst_root / "masks" / "val"

# === Zielverzeichnisse anlegen ===
for path in [dst_img_train, dst_img_val, dst_mask_train, dst_mask_val]:
    path.mkdir(parents=True, exist_ok=True)

# === Dateien sammeln ===
all_images = sorted([p for p in src_images.glob("*.png")])
print(f"Gefundene Bilder: {len(all_images)}")

# === Split berechnen ===
random.shuffle(all_images)
split_idx = int(len(all_images) * split_ratio)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# === Kopieren ===
def copy_pairs(image_list, img_dst, mask_dst):
    for img_path in image_list:
        mask_path = src_masks / img_path.name
        if not mask_path.exists():
            print(f"⚠️ Maske fehlt für: {img_path.name}")
            continue
        shutil.copy(img_path, img_dst / img_path.name)
        shutil.copy(mask_path, mask_dst / mask_path.name)

copy_pairs(train_images, dst_img_train, dst_mask_train)
copy_pairs(val_images, dst_img_val, dst_mask_val)

print(f"✅ Fertig! {len(train_images)} Trainingsbilder, {len(val_images)} Validierungsbilder verteilt.")
