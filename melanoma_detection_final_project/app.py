import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from melanoma_detection_load_model import predict_single_image, transform

# Streamlit app
def main():
    st.title("Melanoma Detection App 🦠")
    # Description of the app
    st.write("""
    **Welcome to the Melanoma Detection App!** A tool designed to assist in assessing skin spots for potential melanoma.
    
    The primary goal of this app is to raise awareness about melanoma and emphasize the importance of routine examinations by a dermatologist.
    Early detection plays a crucial role in effective treatment and improved outcomes.
    
    - 💡 While this app provides valuable insights, it's important to note that **it does not replace professional medical advice.**
    - 📅 Remember to schedule regular visits with a dermatologist for comprehensive skin examinations.
    

    **How to Use:**
    1. **Capture a Close-Up Photo:** Take a clear, close-up image of the suspicious skin spot. Ensure that the spot is well-focused and visible within the frame.
    2. **Upload the Photo:** Use the upload feature to submit the captured image to the app.
    3. **Receive Prediction:** Our model will analyze the uploaded image and estimate the likelihood of the spot being malignant or benign.
    """)

    st.markdown("---")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    st.write("""💡 **Please keep in mind that while our model is highly accurate (up to 95% precision),
    the results might not be precise, and it should be used as a supplementary tool alongside professional medical evaluations.**""")

    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded Image:")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the uploaded image using the transformation from load_model.py
        transformed_image = transform(image)
        st.subheader("Transformed Image:")
        st.image(transformed_image, caption="Transformed Image", use_column_width=True)

        # Make prediction on the uploaded image
        probabilities, predicted_class = predict_single_image(uploaded_file)
        malignant_prob = probabilities[0] * 100
        benign_prob = probabilities[1] * 100

        st.write("###### Prediction Probabilities:")
        st.write(f"- Malignant: {malignant_prob:.2f}%")
        st.write(f"- Benign: {benign_prob:.2f}%")

        # Display prediction result with a color-coded label
        if predicted_class == 0:
            prediction_result = "Malignant"
        else:
            prediction_result = "Benign"

        st.write(f"###### Prediction: {prediction_result}")

        # Display a bar chart for prediction probabilities
        st.write("###### Prediction Visualization:")
        chart_data = {
            'Class': ['Malignant', 'Benign'],
            'Probability': [malignant_prob, benign_prob]
        }
        st.bar_chart(chart_data)
# Run the app
if __name__ == "__main__":
    main()