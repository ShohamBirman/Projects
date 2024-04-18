# load_model.py
import torch
from torchvision import transforms
from PIL import Image
from model import MelanomaClassifier
# Load the saved model weights
model_path = r"C:\Users\shoha\OneDrive\מסמכים\GitHub\Projects\melanoma_detection_final_project\melanoma_classifier.pth"

# Load model on CPU (specify map_location=torch.device('cpu'))
model = MelanomaClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_single_image(image_path):
    """
    Function to predict the class probabilities of a single image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        tuple: Tuple containing predicted probabilities for each class and the predicted class index.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        #probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        probabilities = torch.softmax(output, dim=1).squeeze(0).tolist()  # Convert logits to probabilities
        _, predicted = torch.max(output, 1)  # Get predicted class index
    _, predicted = torch.max(output, 1)  # Get predicted class index

    return probabilities, predicted.item()
    #return probabilities.squeeze(0).tolist(), predicted.item()