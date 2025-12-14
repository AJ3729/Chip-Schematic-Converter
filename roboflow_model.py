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

image = cv2.imread("data/cleaned/circuit_199.jpg")

results = client.infer(
    image,
    model_id="electrical-components-qpl1s/6"
)

detections = sv.Detections.from_inference(results)

# Create annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# Annotate image
annotated_image = image.copy()
annotated_image = box_annotator.annotate(
    scene=annotated_image,
    detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

# Show image
cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
