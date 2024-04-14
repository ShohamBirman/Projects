import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import zipfile
from PIL import Image
import streamlit as st
# Extract the dataset zip file
with zipfile.ZipFile("C:\\Users\\shoha\\OneDrive\\מסמכים\\GitHub\\final_project\\archive.zip", "r") as zip_ref:
    zip_ref.extractall("extracted_dataset")

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define dataset class to load images from extracted directory
class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images and labels
        self._load_data()

    def _load_data(self):
        classes = os.listdir(self.root)
        for idx, class_name in enumerate(classes):
            class_path = os.path.join(self.root, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.images.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# Load datasets
train_dataset = ZipDataset("extracted_dataset/train", transform=transform)
test_dataset = ZipDataset("extracted_dataset/test", transform=transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# Define CNN model
class MelanomaClassifier(nn.Module):
    def __init__(self):
        super(MelanomaClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes: melanoma and benign

    def forward(self, x):
        return self.model(x)


model = MelanomaClassifier()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


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