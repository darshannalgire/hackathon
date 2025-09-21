import os
import json
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import requests

app = Flask(__name__)

# This is a placeholder for your actual AI model.
# In a real-world scenario, you would load your trained TensorFlow or PyTorch model here.
# Example: model = tf.keras.models.load_model('your_trained_model.h5')

# Placeholder for a mock AI model and advice data.
# In a real scenario, this would be a database or a more complex system.
MOCK_DATA = {
    "potato__early_blight": {
        "disease": "Potato Early Blight",
        "confidence": 92.5,
        "advice": "Use fungicides with an active ingredient such as chlorothalonil or mancozeb. Ensure proper irrigation and drainage to prevent excessive leaf wetness."
    },
    "tomato__bacterial_spot": {
        "disease": "Tomato Bacterial Spot",
        "confidence": 88.0,
        "advice": "Remove and destroy infected plant parts. Apply copper-based fungicides or bactericides. Avoid overhead watering to reduce spread."
    },
    "corn__common_rust": {
        "disease": "Corn Common Rust",
        "confidence": 95.1,
        "advice": "Apply fungicides containing azoxystrobin or propiconazole. Consider planting resistant corn hybrids. Ensure good air circulation in the field."
    },
    "healthy": {
        "disease": "Healthy Plant",
        "confidence": 99.8,
        "advice": "Your plant looks healthy! Continue with regular watering and care to maintain its health."
    }
}


# Placeholder function to simulate an AI prediction.
# Replace this function with your actual model inference code.
def predict_disease(image_bytes):
    """
    Simulates a disease prediction based on the uploaded image.
    This function should be replaced with your actual model inference logic.
    """
    # In a real app, you would preprocess the image here and run your model.
    # For example:
    # from tensorflow.keras.preprocessing.image import img_to_array, load_img
    # image = load_img(io.BytesIO(image_bytes), target_size=(256, 256))
    # image_array = img_to_array(image)
    # image_array = np.expand_dims(image_array, axis=0)
    # prediction = model.predict(image_array)

    # For this prototype, we'll just return a mock prediction.
    # We could try to infer from the filename, but for simplicity, we'll pick a random one.
    import random
    return random.choice(list(MOCK_DATA.keys()))


# Placeholder function to get advice from a conversational model.
# You will replace this with a real Gemini API call.
# The payload and response structure are for demonstration.
def get_gemini_advice(disease_name, user_query):
    """
    Simulates a call to the Gemini API to get detailed advice.
    This is where you would integrate the real Google Gemini API.
    """
    try:
        # Define your API payload for a text generation request.
        # This is for demonstration purposes.
        prompt = f"Act as a professional agricultural expert. Provide concise and practical advice for a farmer on how to manage and treat '{disease_name}'. Also, provide information about preventive measures."
        
        # A real Gemini API call would look something like this:
        # payload = {
        #     "contents": [{
        #         "parts": [{"text": prompt}]
        #     }],
        #     "generationConfig": {
        #         "responseMimeType": "text/plain",
        #     }
        # }
        # api_key = os.environ.get("GEMINI_API_KEY")
        # api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        # response = requests.post(api_url, json=payload)
        # response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        # return response.json()['candidates'][0]['content']['parts'][0]['text']

        # For this prototype, we'll return mock data.
        return MOCK_DATA.get(disease_name, {}).get("advice", "No specific advice available for this disease.")
    except Exception as e:
        print(f"Error calling Gemini API placeholder: {e}")
        return "An error occurred while fetching advice. Please try again later."


@app.route('/')
def home():
    """Renders the main page of the application."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and returns a disease prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            image_bytes = file.read()
            
            # Step 1: Predict the disease using the mock model.
            predicted_class = predict_disease(image_bytes)
            
            # Step 2: Get detailed advice based on the prediction.
            advice = get_gemini_advice(predicted_class, user_query="")
            
            # Step 3: Get the confidence and disease name from our mock data.
            result = MOCK_DATA.get(predicted_class, MOCK_DATA["healthy"])
            
            response = {
                "disease": result["disease"],
                "confidence": f"{result['confidence']:.2f}%",
                "advice": advice
            }
            
            return jsonify(response)

        except Exception as e:
            app.logger.error(f"Prediction failed: {e}")
            return jsonify({"error": "Prediction failed. Please try again."}), 500

    return jsonify({"error": "Something went wrong"}), 500


if __name__ == '__main__':
    # You should set FLASK_ENV to 'development' for debugging.
    # In a production environment, use a production-ready server like Gunicorn.
    # Example: gunicorn --bind 0.0.0.0:5000 app:app
    app.run(debug=True, host='0.0.0.0', port=5000)
