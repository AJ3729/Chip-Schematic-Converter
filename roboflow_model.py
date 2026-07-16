import cv2
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import supervision as sv
import json

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

image_path = "data/cleaned/circuit_1199.jpg"
image = cv2.imread(image_path)

results = client.infer( 
    image,
    model_id="electrical-components-qpl1s/6"
)

detections = sv.Detections.from_inference(results)

# ---- SAVE DETECTIONS TO JSON ----
detections_list = []

for pred in results["predictions"]:
    detections_list.append({
        "class": pred["class"],
        "confidence": pred["confidence"],
        "x": pred["x"],
        "y": pred["y"],
        "width": pred["width"],
        "height": pred["height"]
    })

output_data = {
    "image": os.path.basename(image_path),
    "detections": detections_list
}

with open("detections.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("Saved detections to detections.json")

# ---- VISUALIZATION (UNCHANGED) ----

# Create annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

annotated_image = image.copy()

annotated_image = box_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()