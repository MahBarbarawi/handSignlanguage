from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from AI import x
AiProcess = x()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')


def image_to_base64(image):
    # Convert the image to PIL Image object
    image_pil = Image.fromarray(image)
    # Create an in-memory buffer to hold the image data
    buffer = BytesIO()
    # Save the image to the buffer in JPEG format
    image_pil.save(buffer, format='JPEG')
    # Convert the buffer's contents to a byte string
    image_bytes = buffer.getvalue()
    # Encode the byte string as base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64

@app.route('/process-video', methods=['POST'])
def process_video():
    image_data = request.json['image']
    # Process the video frame as needed
    processed_image = process_frame(image_data)
    processed_image_data = image_to_base64(processed_image)
    # Return the result as a JSON response
    return jsonify({'image': processed_image_data})

def process_frame(image_data):
    # Perform any necessary processing on the video frame
    # For example, you can save it to a file or analyze it
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))

    # Perform any necessary processing on the image
    # For example, you can apply filters, resize, or manipulate the image

    # Convert the processed image to a NumPy array
    processed_image = np.array(image)
    print(type(processed_image))
    print("image_data", processed_image.shape)
    image_data = AiProcess.PicProce(processed_image)
    # cv2.imshow("ww",image_data)
    return image_data

if __name__ == '__main__':
    app.run()
