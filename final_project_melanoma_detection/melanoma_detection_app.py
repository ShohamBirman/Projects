import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from Melanoma_Detection import MelanomaClassifier

# Define transform for preprocessing input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your trained model
model = MelanomaClassifier()
model.load_state_dict(torch.load(r'C:\Users\shoha\OneDrive\מסמכים\GitHub\final_project\model.pth'))
model.eval()


# Define a function to make predictions
@st.cache
def predict(image):
    try:
        image = Image.open(image).convert("RGB")
    except:
        st.error('Please upload a valid image file.')
        return

    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()  # Return the predicted class index (0 or 1)


# Streamlit app
def main():
    st.title('Melanoma Detection App')
    st.text('Upload a skin image to check for melanoma')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect Melanoma'):
            prediction = predict(uploaded_file)
            if prediction == 1:
                st.error('Warning: Melanoma detected!')
            else:
                st.success('No signs of melanoma detected.')


if __name__ == '__main__':
    main()
