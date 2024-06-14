from flask import Flask, render_template, Response,request,jsonify
import base64
import io
import numpy as np
from PIL import Image
import pickle as pk
from AI import x
import os
import time
from datetime import datetime

# Get the current time
with open ("./ArabicTranslation.pkl" , "rb") as p :
    arabicDic = pk.load(p)
import cv2
AiProcess = x()
app = Flask(__name__)
from PIL import Image

global i
i=0
# @app.route('/upload', methods=['POST'])
# def upload():
#     # image_data = request.form['image_data']
#     current_time = datetime.now()
#     image_data = request.form['image_data']
#
#     # Extract the base64-encoded image data
#     image_data = image_data.split(",")[1]
#
#     # Decode the base64-encoded image data
#     image_bytes = base64.b64decode(image_data)
#
#     # Convert the image bytes to a PIL Image object
#     image = Image.open(io.BytesIO(image_bytes))
#
#     # Convert the PIL Image to a NumPy array
#     image_array = np.array(image)
#     print(image_array.shape)
#     # Reshape the image to a different width and height
#     new_width = 640
#     new_height = 480
#
#     rgb_array =  cv2.resize(image_array[:, :, :3], (new_width, new_height))
#     # print("a" ,image_array)
#     # print("Done2")
#     # print(type(rgb_array))
#     # np.save("./Data/"+str(i)+".npy",image_array)
#     # print(rgb_array)
#     data =AiProcess.PicProce(rgb_array)
#     print(data)
#     print(rgb_array.shape)
#     hours = current_time.hour
#     minutes = current_time.minute
#     seconds = current_time.second
#     milliseconds = current_time.microsecond // 1000
#     image_path = os.path.join("./Data/",str(hours)+str(minutes)+str(seconds)+str(milliseconds)+".png" )
#     image.save(image_path)
#     cv2.imshow("aa",image_array)
#     return data
#     # Perform further processing on the image if needed

# @app.route('/upload', methods=['POST'])
# def upload():
#     print("!11")
#     # Check if the 'image_data' field exists in the request
#     if request.is_json:
#         print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#         image_data = request.json['frame']
#         image_bytes = base64.b64decode(image_data.split(',')[1])
#         current_time = datetime.now()
#         # Convert image bytes to PIL Image object
#         image = Image.open(io.BytesIO(image_bytes))
#
#         # Resize the image to 480x640 pixels
#         # resized_image = image.resize((640, 480))
#         hours = current_time.hour
#         minutes = current_time.minute
#         seconds = current_time.second
#         milliseconds = current_time.microsecond // 1000
#         image_path = os.path.join("./Data/",str(hours)+str(minutes)+str(seconds)+str(milliseconds)+".png" )
#         image.save(image_path)
#         np_array = np.array(image)
#         data = AiProcess.PicProce(np_array[:,:,:3])
#         response_data = {
#             'message': data
#         }
#         return jsonify(response_data)
#     else:
#         return 'No image data received'

@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_data = request.json['frame']
    image_bytes = base64.b64decode(frame_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))

    np_array = np.array(image)
    # print(np_array.shape)
    data=AiProcess.PicProce(np_array)
    print(arabicDic[data])
    translation = arabicDic[data]
    return jsonify({"translation": translation})
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/teaching')
def teaching():
    return render_template('teaching.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

