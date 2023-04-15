from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import *
from fastai.vision.all import *

app = Flask(__name__)
model_path = "model.pkl"
model = load_learner(model_path, cpu=True)

@app.route('/', methods=['GET'])
def main_route():
    return jsonify({'message': "carification app"})

@app.route('/predict_image', methods=['POST'])
def process_image():
    try:
        file = request.files['image']
        img = Image.open(file.stream)
        img = img.resize((299,299))
        prediction,_,_ = model.predict(img)
        return jsonify({'prediction': prediction})
    except (BadRequest, KeyError):
        return jsonify({'error': "bad request"})
    except UnidentifiedImageError:
        return jsonify({'error': "file is not a valid image"})
    except Exception:
        return jsonify({'error': "error on prediction data"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)