from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from waitress import serve
from werkzeug.exceptions import *
from fastai.vision import *
from fastai.vision.all import *

app = Flask(__name__)
model_path = "224_resnet50_unfreeze_da_dlrs_lra_mult.pkl"
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
        return prediction
    except (BadRequest, KeyError):
        return jsonify({'error': "bad request"})
    except UnidentifiedImageError:
        return jsonify({'error': "file is not a valid image"})
    except Exception:
        return jsonify({'error': "error on prediction data"})

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)