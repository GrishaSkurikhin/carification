import configparser
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from waitress import serve
from werkzeug.exceptions import *
from model import Model

app = Flask(__name__)
config = configparser.ConfigParser()
config.read('config.ini')
model = Model(config["model"]["path"])

@app.route('/predict_image', methods=['POST'])
def process_image():
    try:
        file = request.files['image']
        img = Image.open(file.stream)
    except (BadRequest, KeyError):
        return jsonify({'error': "bad request"})
    except UnidentifiedImageError:
        return jsonify({'error': "file is not a valid image"})
    except Exception:
        return jsonify({'error': "error on receiving and processing data"})
    
    try:
        prediction = model.predict(img)
        return jsonify({'prediction': prediction})
    except Exception:
        return jsonify({'error': "prediction error"})

if __name__ == '__main__':
    serve(app, host=config["server"]["host"], port=config["server"]["port"])