from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("model.h5", compile=False)

# Classes as per your model
classes = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data)).resize((224, 224)).convert("RGB")
        image_array = np.array(image)
        image_array = image_array / 255.0  # normalize
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_class = classes[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
