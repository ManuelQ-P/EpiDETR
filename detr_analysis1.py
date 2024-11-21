import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T

# Load the pre-trained DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# COCO class names
COCO_CLASSES = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Function to transform frames for DETR
def transform_frame(frame):
    transform = T.Compose([T.Resize(800), T.ToTensor()])
    return transform(frame).unsqueeze(0)

# Function to scale bounding boxes to the original frame size
def rescale_bboxes(boxes, original_size):
    orig_w, orig_h = original_size
    boxes = boxes * torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
    return boxes.cpu().numpy()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Transform the frame for the DETR model
    inputs = transform_frame(pil_image)
    original_size = pil_image.size

    # Run the frame through the model
    with torch.no_grad():
        outputs = model(inputs)

    # Extract the bounding boxes and labels
    probabilities = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]

    # Rescale bounding boxes to the original frame size
    scaled_boxes = rescale_bboxes(boxes, original_size)

    # Draw the bounding boxes and labels on the frame
    for p, (xmin, ymin, xmax, ymax) in zip(probabilities, scaled_boxes):
        cl = p.argmax()
        score = p[cl].item()
        if score < 0.7:  # Adjust the confidence threshold as needed
            continue
        label = COCO_CLASSES[cl] if cl < len(COCO_CLASSES) else "Unknown"

        # Draw the bounding box
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        # Put the label and confidence score
        cv2.putText(frame, f'{label}: {score:.2f}', (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('DETR Real-Time Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


