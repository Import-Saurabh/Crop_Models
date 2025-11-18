from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

print("Current working directory:", os.getcwd())
print("Templates path exists:", os.path.exists('templates'))
print("Templates index.html exists:", os.path.exists('templates/index.html'))

# -----------------------------------------------------
# LOAD MAIZE MODEL
# -----------------------------------------------------
maize_model = None
maize_le = None
maize_mlb = None

try:
    maize_artifacts = joblib.load('maize_disease_model.pkl')
    maize_model = maize_artifacts['model']
    maize_le = maize_artifacts['label_encoder']
    maize_mlb = maize_artifacts['mlb']
    print("✅ Maize model loaded.")
except Exception as e:
    print("❌ Maize model load failed:", e)


# -----------------------------------------------------
# LOAD WHEAT MODEL (single-label)
# -----------------------------------------------------
wheat_model = None
wheat_le = None
wheat_mlb = None

try:
    wheat_artifacts = joblib.load('wheat_disease_model.pkl')
    wheat_model = wheat_artifacts['model']
    wheat_le = wheat_artifacts['label_encoder']
    wheat_mlb = wheat_artifacts['mlb']
    print("✅ Wheat model loaded.")
except Exception as e:
    print("❌ Wheat model load failed:", e)


# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -----------------------------------------------------
# GET CROP STAGES (JSON serializable)
# -----------------------------------------------------
@app.route('/crop_stages', methods=['GET'])
def crop_stages():
    crop = request.args.get('crop_type', '').lower()

    # ----------------- MAIZE -----------------
    if crop == "maize":
        if maize_le is None:
            return jsonify({'crop_stages': []})

        # FIX: Convert numpy int64 → python int
        stages = [int(x) for x in maize_le.classes_]
        return jsonify({'crop_stages': stages})

    # ----------------- WHEAT -----------------
    if crop == "wheat":
        if wheat_le is None:
            return jsonify({'crop_stages': []})

        stages = [int(x) for x in wheat_le.classes_]
        return jsonify({'crop_stages': stages})

    return jsonify({'crop_stages': []})


# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    crop = data.get('crop_type')
    stage = data.get('crop_stage')

    if stage is None:
        return jsonify({'error': 'Crop stage required'}), 400

    # Convert stage to python int for LabelEncoder
    try:
        stage = int(stage)
    except:
        return jsonify({'error': 'Invalid stage value'}), 400

    # ----------------- MAIZE -----------------
    if crop == "maize":
        if maize_model is None:
            return jsonify({'error': 'Maize model not loaded'}), 500

        encoded = maize_le.transform([stage])
        pred = maize_model.predict(np.array(encoded).reshape(-1, 1))
        diseases = maize_mlb.inverse_transform(pred)[0]

        return jsonify({
            'status': 'success',
            'predicted_diseases': list(diseases)
        })

    # ----------------- WHEAT -----------------
    if crop == "wheat":
        if wheat_model is None:
            return jsonify({'error': 'Wheat model not loaded'}), 500

        encoded = wheat_le.transform([stage])
        pred = wheat_model.predict(np.array(encoded).reshape(-1, 1))
        diseases = wheat_mlb.inverse_transform(pred)[0]

        return jsonify({
            'status': 'success',
            'predicted_diseases': list(diseases)
        })

    return jsonify({'error': 'Invalid crop type'}), 400


# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'maize_model_loaded': maize_model is not None,
        'wheat_model_loaded': wheat_model is not None,
        'working_directory': os.getcwd()
    })


if __name__ == "__main__":
    print("\n🌱 Starting Crop Disease Prediction Server...")
    print("📍 http://localhost:5000")
    app.run(debug=True)
