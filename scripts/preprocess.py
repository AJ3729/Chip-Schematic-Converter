import cv2
import numpy as np
import os
from tqdm import tqdm

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(path):
    img = cv2.imread(path)

    # Convert to grayscale early
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary for rotation analysis
    _, binary_for_angle = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours for rotation detection
    coords = np.column_stack(np.where(binary_for_angle > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Convert to grayscale after rotation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove shadows
    dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

    # Binarize
    _, binary = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small specks using contour area filtering
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 50:  # adjust threshold for speck size
            cv2.drawContours(binary, [c], -1, 0, -1)

    # --- Auto-crop to content ---
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]

    # Resize
    cleaned = cv2.resize(cropped, (512, 512))

    return cleaned



if __name__ == "__main__":
    images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    for img_name in tqdm(images, desc="Preprocessing images"):
        img_path = os.path.join(RAW_DIR, img_name)
        processed = preprocess_image(img_path)
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), processed)
