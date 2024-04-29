# load_model.py
import torch
from torchvision import transforms
from PIL import Image
from model_efficientnet import MelanomaClassifier

# Load the saved model weights
model_path = "melanoma_detection_final_project/EfficientNet/melanoma_classifier_weights_EfficientNet.pth"

# Load model on CPU (specify map_location=torch.device('cpu'))
model = MelanomaClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
test_transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize images to 112x112
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_single_image(file_obj):
    """
    Function to predict the class probabilities of a single image.
    Args:
        file_obj (BytesIO): File object containing the image data.
    Returns:
        tuple: Tuple containing predicted probabilities for each class and the predicted class index.
    """
    # Load and preprocess the image
    image = Image.open(file_obj).convert("RGB")
    image = test_transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).squeeze(0).tolist()  # Convert logits to probabilities
        _, predicted = torch.max(output, 1)  # Get predicted class index

    return probabilities, predicted.item()
