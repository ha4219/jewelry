from flask import Flask, redirect, render_template, request, url_for
from graphviz import render
from werkzeug.utils import secure_filename
import os
import sys
from datetime import datetime
import torch
# for import face_alignment
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "Face-alignment"))
import face_alignment
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "Face-parsing"))
# below two modules in face_parsing directory
from face_parsing import execute_face_parsing
from face_parsing_model import BiSeNet

from skimage import io
import numpy as np
app = Flask(__name__)

# SET PATH
IMG_DIR_PATH = os.path.join('assets', 'images')
ALIGNMENT_DIR_PATH = os.path.join('assets', 'face_alignment_results')
PARSING_DIR_PATH = os.path.join('assets', 'face_parsing_results')
WEIGHTS_PATH = "../Face-parsing/res/cp/79999_iter.pth"

# Model for Face Alignment
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

# Model for Face Parsing
NET = BiSeNet(19)
NET.cuda()
NET.load_state_dict(torch.load(WEIGHTS_PATH))
NET.eval()

def execute_face_alignment(img_path, dst_path):
    input_ = io.imread(img_path)
    preds = FA.get_landmarks(input_)
    np.array(preds).tofile(dst_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/complete')
def process_complete():
    return render_template('complete.html')

@app.route('/uploader', methods=['POST'])
def uploader_file():
    if request.method == "POST":
        f = request.files['file']

        current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        image_path = os.path.join(IMG_DIR_PATH, secure_filename(current_time+".jpg"))
        fa_dst_path = os.path.join(ALIGNMENT_DIR_PATH, current_time+'.bin')
        fp_dst_path = os.path.join(PARSING_DIR_PATH, current_time+'.png')
        # Save Image in file_path
        f.save(image_path)
        
        # face_alignment
        execute_face_alignment(image_path, fa_dst_path)
        # face_parsing
        execute_face_parsing(fp_dst_path, image_path, NET)

        return redirect(url_for('process_complete'))

if __name__ == '__main__':
    app.run(debug=True)