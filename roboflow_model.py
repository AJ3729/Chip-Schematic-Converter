import json
from inference import get_model
import supervision as sv
import cv2

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

print(f"API Key: {api_key}")
print(f"API Key is None: {api_key is None}")

os.makedirs('results', exist_ok=True)

# define the image url to use for inference
image_file = "data/cleaned/circuit_1199.jpg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="electrical-components-qpl1s/3")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

with open('detections.json', 'w') as f:
    json.dump(results.dict(), f, indent=4)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
output_path = 'output.jpg'
success = cv2.imwrite(output_path, annotated_image)

if success:
    print(f"✅ Image saved successfully to {output_path}")
else:
    print("❌ Failed to save image")
print("Results saved to detections.json and output.jpg")