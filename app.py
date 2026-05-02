import io
import os
import datetime
import logging

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Layers ---
import tensorflow as tf
import keras
from keras import layers

@keras.saving.register_keras_serializable()
class ReduceMeanLayer(layers.Layer):
    def call(self, x):
        import keras.ops as kops
        return kops.mean(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable()
class ReduceMaxLayer(layers.Layer):
    def call(self, x):
        import keras.ops as kops
        return kops.max(x, axis=-1, keepdims=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)

IMG_SIZE = (128, 128)
NUM_CLASSES = 4
CLASS_NAMES = ['Broadleaf Weed', 'Grass Weed', 'Soil', 'Soybean']

# Relaxed thresholds
DEMO_MIN_FIELD_LIKENESS = 0.05
LIVE_MIN_CONFIDENCE = 0.30

CLASS_INFO = {
    'Soil': {'color': '#8B6914', 'risk': 'none', 'action': 'Healthy bare soil detected.'},
    'Soybean': {'color': '#4CAF50', 'risk': 'none', 'action': 'Healthy soybean crop.'},
    'Grass Weed': {'color': '#FF9800', 'risk': 'medium', 'action': 'Grass weed detected.'},
    'Broadleaf Weed': {'color': '#F44336', 'risk': 'high', 'action': 'Broadleaf weed detected.'},
}

_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    try:
        model_path = 'mt_ahnet_model.keras'
        if os.path.exists(model_path):
            _model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'ReduceMeanLayer': ReduceMeanLayer, 'ReduceMaxLayer': ReduceMaxLayer}
            )
            print(f"INFO: Model loaded from {model_path}")
        else:
            print(f"WARNING: {model_path} not found.")
    except Exception as exc:
        print(f"ERROR: Model load failed: {exc}")
    return _model

@app.route('/')
@app.route('/index.html')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/stats')
def stats():
    return jsonify({
        'model': {'accuracy': 0.942, 'precision': 0.928, 'recall': 0.915, 'f1_score': 0.921, 'val_loss': 0.142},
        'dataset': {'classes': {'Soybean': {'count': 7376}, 'Soil': {'count': 3249}, 'Grass Weed': {'count': 3520}, 'Broadleaf Weed': {'count': 1191}}}
    })

@app.route('/api/health')
def health():
    m = get_model()
    return jsonify({
        'status': 'ok',
        'model_loaded': m is not None,
        'classes': CLASS_NAMES,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    f = request.files['image']
    data = f.read()
    
    # Pre-check likeness
    score = 0.0
    try:
        img = Image.open(io.BytesIO(data)).convert('HSV')
        hsv = np.array(img, dtype=np.uint8)
        v = hsv[..., 2].astype(np.float32)
        valid = v > 20
        valid_count = max(1, int(valid.sum()))
        h = hsv[..., 0].astype(np.float32)
        s = hsv[..., 1].astype(np.float32)
        green = ((h >= 35) & (h <= 110) & (s > 30) & (v > 30) & valid)
        yellow = ((h >= 18) & (h <= 45) & (s > 30) & (v > 50) & valid)
        brown = ((h >= 8) & (h <= 28) & (s > 20) & (v > 20) & valid)
        score = float((green.sum() + yellow.sum() + brown.sum()) / valid_count)
    except:
        pass

    print(f"DEBUG: Likeness Score = {score:.4f}")
    
    if score < DEMO_MIN_FIELD_LIKENESS:
        return jsonify({'error': f'Rejected (Score: {score:.2f}). Try a clearer field image.'}), 422

    model = get_model()
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503
        
    # Preprocess
    img_rgb = Image.open(io.BytesIO(data)).convert('RGB').resize(IMG_SIZE)
    x = np.expand_dims(np.array(img_rgb, dtype=np.float32) / 255.0, axis=0)
    
    # Predict
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    
    return jsonify({
        'predicted_class': CLASS_NAMES[idx],
        'confidence': confidence,
        'probabilities': {CLASS_NAMES[i]: float(preds[i]) for i in range(NUM_CLASSES)},
        'class_info': CLASS_INFO.get(CLASS_NAMES[idx], {}),
        'demo_mode': False
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting MT-AHNet Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
