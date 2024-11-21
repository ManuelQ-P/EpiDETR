import torch
import sys
from PIL import Image
import requests
import matplotlib.pyplot as plt
from torchvision import transforms as T
import io

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

# Load and preprocess the image
def load_image(image_path):
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    
    # Get the original size of the image
    original_size = image.size

    # Transform the image to the model's input size (800 pixels)
    transform = T.Compose([T.Resize(800), T.ToTensor()])
    resized_image = transform(image).unsqueeze(0)
    
    return resized_image, image, original_size

# Function to scale bounding boxes to the original image size
def rescale_bboxes(boxes, original_size):
    orig_w, orig_h = original_size
    boxes = boxes * torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
    boxes = boxes.cpu().numpy()
    return boxes

# Visualize the results
def plot_results(image, probas, boxes, confidence_threshold):
    plt.imshow(image)
    ax = plt.gca()
    summary = []

    # Rescale bounding boxes to the original image size
    original_size = image.size
    scaled_boxes = rescale_bboxes(boxes, original_size)

    for p, (xmin, ymin, xmax, ymax) in zip(probas, scaled_boxes):
        cl = p.argmax()
        score = p[cl].item()
        if score < confidence_threshold:
            continue
        label = COCO_CLASSES[cl] if cl < len(COCO_CLASSES) else "Unknown"
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        text = f'{label}: {score:.2f}'
        ax.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5))
        summary.append(text)

    plt.axis('off')
    plt.show()
    
    # Print a summary of detections
    if summary:
        print("Detections Summary:")
        for item in summary:
            print(item)
    else:
        print("No detections above the confidence threshold.")

# Main function
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python detr_analysis.py <image_path_or_url> <confidence_threshold>")
        sys.exit(1)

    image_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2])
    inputs, original_image, original_size = load_image(image_path)

    # Run the image through the model
    with torch.no_grad():
        outputs = model(inputs)

    # Extract the bounding boxes and labels
    probabilities = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]

    # Display all predictions with scores
    plot_results(original_image, probabilities, boxes, confidence_threshold)



