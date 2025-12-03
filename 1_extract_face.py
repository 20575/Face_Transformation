# 1_extract_face.py - Automatic high-quality face detection & cropping
import cv2
import os
from insightface.app import FaceAnalysis
from pathlib import Path

# Create output folders
os.makedirs("original_photos", exist_ok=True)
os.makedirs("extracted_faces", exist_ok=True)

# Initialize InsightFace model (downloads ~300 MB the first time only)
print("Downloading InsightFace model (only once)...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model ready!")

# Wait for you to put your photos
print("\nPut your selfies in the folder 'original_photos' then press Enter")
input("Press Enter when ready...")

# Process every image in the folder
input_folder = Path("original_photos")
image_files = list(input_folder.glob("*.*"))

if not image_files:
    print("No photos found in 'original_photos' folder!")
else:
    for img_path in image_files:
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
            print(f"Processing: {img_path.name}")
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"   Could not read {img_path.name}")
                continue
                
            faces = app.get(image)
            
            if len(faces) == 0:
                print("   No face detected")
                continue
                
            # Take the biggest face (usually yours)
            main_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            bbox = main_face.bbox.astype(int)
            
            # Add generous padding (keeps forehead, chin, neck)
            padding = 80
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding - 50)   # extra top for forehead/hair
            x2 = min(image.shape[1], bbox[2] + padding)
            y2 = min(image.shape[0], bbox[3] + padding + 100)  # extra bottom for neck
            
            cropped_face = image[y1:y2, x1:x2]
            
            output_path = f"extracted_faces/{img_path.stem}_face.png"
            cv2.imwrite(output_path, cropped_face)
            print(f"   Face saved → {output_path}")

print("\nALL FACES EXTRACTED SUCCESSFULLY!")
