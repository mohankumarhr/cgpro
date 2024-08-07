from flask import Flask, request, render_template
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
import cv2
import base64
from net import Net

app = Flask(__name__)
ML_MODEL = None
ML_MODEL_FILE = "model.pt"
TORCH_DEVICE = "cpu"



def get_model():
    
    global ML_MODEL
    if not ML_MODEL:
        ML_MODEL = Net()
        ML_MODEL.load_state_dict(
            torch.load(ML_MODEL_FILE, map_location=torch.device(TORCH_DEVICE))
        )

    return ML_MODEL

def freshness_label(freshness_percentage):
    if freshness_percentage > 90:
        return "Fresh"
    elif freshness_percentage > 65:
        return "Good"
    elif freshness_percentage > 50:
        return "Good Enough"
    elif freshness_percentage > 0:
        return "Not Good"
    else:
        return "Rotten"



def freshness_percentage_by_cv_image(cv_image):
   
    mean = (0.7369, 0.6360, 0.5318)
    std = (0.3281, 0.3417, 0.3704)
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    image_tensor = transformation(image)
    batch = image_tensor.unsqueeze(0)
    out = get_model()(batch)
    s = nn.Softmax(dim=1)
    result = s(out)
    return int(result[0][0].item()*100)

def imdecode_image(image_file):
    return cv2.imdecode(
        np.frombuffer(image_file.read(), np.uint8),
        cv2.IMREAD_UNCHANGED
    )

def recognize_fruit_by_cv_image(cv_image):
    freshness_percentage = freshness_percentage_by_cv_image(cv_image)
    return {
        "freshness_level": freshness_percentage,
    }



@app.route('/api/recognize', methods=["POST"])
def api_recognize():
    cv_image = imdecode_image(request.files["image"])
    return recognize_fruit_by_cv_image(cv_image)


@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/home")
def home_page():
    return render_template("home.html")


@app.route("/prediction", methods=["POST"])
def prediction_page():
    cv_image = imdecode_image(request.files["image"])
    fruit_information = recognize_fruit_by_cv_image(cv_image)
   
    freshness_percentage = fruit_information["freshness_level"]

 
    image_content = cv2.imencode('.jpg', cv_image)[1].tobytes()
    encoded_image = base64.encodebytes(image_content)
    base64_image = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template(
        "prediction.html",
        freshness_percentage=freshness_percentage,
        freshness_label=freshness_label(freshness_percentage),
        base64_image=base64_image,
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)