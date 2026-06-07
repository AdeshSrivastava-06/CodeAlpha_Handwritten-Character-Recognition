# Handwritten Character Recognition

A Streamlit app that recognizes a single handwritten character from either a freehand drawing or an uploaded image. The model loads a trained CNN from the `model/` folder and returns the top prediction with the top 5 class probabilities.

## Features

- Draw a character directly in the browser and run inference.
- Upload a PNG, JPG, or JPEG image for recognition.
- Display the predicted character with confidence scores.
- Show the top 5 model predictions for quick comparison.

## Tech Stack

- Streamlit
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- streamlit-drawable-canvas

## Project Structure

- `app1.py` - main Streamlit application
- `model/best_model.h5` - trained CNN model
- `model/class_labels.json` - class index to label mapping
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python runtime version for deployment
- `emnist_source_files/` and the EMNIST CSV files - training assets used while building the model

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Run the app:

	```bash
	streamlit run app1.py
	```

## How To Use

1. Open the app in your browser.
2. Use the Draw tab to sketch a single handwritten character, then click Predict.
3. Use the Upload tab to select an image and click Predict.

## Deployment Notes

- Use `app1.py` as the Streamlit entry point.
- Keep the `model/` folder in the repository so the app can load the trained model at runtime.
- The raw EMNIST source files and CSVs are training data and are not required to run the app.
- If you want a lighter GitHub repository, exclude the large EMNIST dataset files or store them with Git LFS.

## Model Details

- Dataset: EMNIST Balanced
- Classes: 47
- Reported accuracy: 90.28%
