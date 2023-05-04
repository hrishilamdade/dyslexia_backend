from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow
from skimage import transform
from flask_cors import CORS
from utils import *
import pickle

app = Flask(__name__)
CORS(app)

#Dyslexia Detection Model
@app.route('/handwriting', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = file.filename
    if filename == '':
        return jsonify({'error': 'No file selected'}), 400
    

    file.save("files/"+filename)


    model = tensorflow.keras.models.load_model("CNN1.hdf5")

    np_image = Image.open( "files/" + filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (28, 28, 3))
    np_image = np.expand_dims(np_image, axis=0)
    # print(training_set.class_indices)
    res = model.predict(np_image)
    print(res)
    #detecting dyslexia
    if res[0][0] < res[0][2] and res[0][1] < res[0][2]:
        return jsonify({'msg': 'Dyslexia Detected'}), 200
    else:
        return jsonify({'msg': 'No Dyslexia Detected'}), 200

#Dyslexia Detection Model from video
@app.route('/video', methods=['POST'])
def upload_video():

    
    mode = request.form["mode"]
    print(mode)
    if mode == "1":
        finDF,dummy_df = get_eyes_points()
    elif mode == "2":
        print(request.files)
        print(request.form)
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filename = file.filename
        if filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file.save("files/"+filename)
        finDF,dummy_df = test_with_txt("files/"+filename)
    model_f = open("rocketClassifier.pkl", "rb")
    # print(model_f)
    model = pickle.load(model_f)
    # print("model loaded\n\n\n\n\n")
    # print(finDF)
    # print(dummy_df)
    res = model.predict([finDF,dummy_df])
    print(res)
    #detecting dyslexia
    if res[0] == 1:
        return jsonify({'msg': 'Dyslexia Detected'}), 200
    else:
        return jsonify({'msg': 'No Dyslexia Detected'}), 200
    

if __name__ == '__main__':
    app.run(debug=True)
