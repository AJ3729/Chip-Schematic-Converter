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

# def preprocess_image(path):
#     img = cv2.imread(path)

#     # Convert to grayscale early
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Threshold to get binary for rotation analysis
#     _, binary_for_angle = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Find contours for rotation detection
#     coords = np.column_stack(np.where(binary_for_angle > 0))
#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle

#     # Rotate image
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     img = cv2.warpAffine(img, M, (w, h),
#                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     # Convert to grayscale after rotation
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Remove shadows
#     dilated_img = cv2.dilate(gray, np.ones((7,7), np.uint8))
#     bg_img = cv2.medianBlur(dilated_img, 21)
#     diff_img = 255 - cv2.absdiff(gray, bg_img)
#     norm_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

#     # Binarize
#     _, binary = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Remove small specks using contour area filtering
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in contours:
#         if cv2.contourArea(c) < 50:  # adjust threshold for speck size
#             cv2.drawContours(binary, [c], -1, 0, -1)

#     # --- Auto-crop to content ---
#     coords = cv2.findNonZero(binary)
#     x, y, w, h = cv2.boundingRect(coords)
#     cropped = binary[y:y+h, x:x+w]

#     # Resize
#     cleaned = cv2.resize(cropped, (512, 512))

#     return cleaned

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None

    H0, W0 = img.shape[:2]

    # --- grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Slight blur to stabilize thresholding
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # --- fast foreground mask for angle estimation (ink = 1) ---
    bin_inv = cv2.threshold(gray_blur, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Erode to suppress hairline page borders before angle detection
    bin_inv_er = cv2.erode(bin_inv, np.ones((3, 3), np.uint8), iterations=1)

    # Use connected components; pick largest interior blob for angle
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_inv_er, connectivity=8)
    # stats: [label, x, y, w, h, area]; label 0 is background
    angle = 0.0
    if num > 1:
        # drop tiny specks & page-like boxes hugging the frame
        H, W = bin_inv_er.shape
        candidates = []
        for i in range(1, num):
            x, y, w, h, a = stats[i]
            if a < 200:                 # specks
                continue
            if x <= 1 or y <= 1 or x+w >= W-2 or y+h >= H-2:
                # touches frame → likely border noise
                continue
            candidates.append((a, i))
        if candidates:
            # largest interior component
            _, idx = max(candidates, key=lambda t: t[0])
            mask = (labels == idx).astype(np.uint8) * 255
            coords = np.column_stack(np.where(mask > 0))
            rect = cv2.minAreaRect(coords)
            raw_angle = rect[-1]
            angle = -(90 + raw_angle) if raw_angle < -45 else -raw_angle

    # --- rotate image by estimated angle ---
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Ensure landscape (optional but helps with sideways pages)
    if img.shape[0] > img.shape[1] * 1.05:
        # rotate 90° CCW to make width > height
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # --- grayscale after rotation ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Conditional shadow normalization (skip if page already clean)
    if gray.std() > 18:  # heuristic: only normalize if lighting is uneven
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    else:
        norm = gray

    # --- binarize (black ink on white) ---
    binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # --- remove specks via connected components (more robust than contours) ---
    num, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary, connectivity=8)
    # We inverted so components are the dark strokes.
    keep = np.zeros_like(binary)
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < 120:                 # tune: minimum stroke area
            continue
        if (w <= 2 and h <= 2):     # single-pixel dust
            continue
        keep[y:y+h, x:x+w][labels[y:y+h, x:x+w] == i] = 255
    binary = 255 - keep  # back to white=255, black=0

    # --- tight crop to content while ignoring edge junk ---
    fg = (binary == 0).astype(np.uint8) * 255  # foreground = black ink
    # close small gaps so the crop is solid
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    coords = cv2.findNonZero(fg)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # add a small padding, clip to image
        pad = 12
        x = max(0, x - pad); y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2*pad)
        h = min(img.shape[0] - y, h + 2*pad)
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary  # fallback

    # --- resize WITH aspect ratio preserved onto 512x512 canvas ---
    target = 512
    ch, cw = cropped.shape[:2]
    scale = min(target / ch, target / cw)
    nh, nw = int(round(ch * scale)), int(round(cw * scale))
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((target, target), 255, dtype=np.uint8)  # white
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized

    return canvas




if __name__ == "__main__":
    images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    for img_name in tqdm(images, desc="Preprocessing images"):
        img_path = os.path.join(RAW_DIR, img_name)
        processed = preprocess_image(img_path)
        cv2.imwrite(os.path.join(CLEAN_DIR, img_name), processed)

