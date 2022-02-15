from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sys
from datetime import datetime
# for import face_alignment
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "Face-alignment"))
import face_alignment
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "Face-parsing"))
from face_parsing import vis_parsing_maps, evaluate
from skimage import io
import numpy as np
app = Flask(__name__)

# SET PATH
IMG_DIR_PATH = os.path.join('assets', 'images')
ALIGNMENT_DIR_PATH = os.path.join('assets', 'face_alignment_results')
PARSING_DIR_PATH = os.path.join('assets', 'face_parsing_results')
WEIGHTS_PATH = "../Face-parsing/res/cp/79999_iter.pth"

def execute_face_alignment(img_path, dst_path):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    input_ = io.imread(img_path)
    preds = fa.get_landmarks(input_)
    np.array(preds).tofile(".\\" + dst_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    if request.method == "POST":
        f = request.files['file']

        current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_path = os.path.join(IMG_DIR_PATH, secure_filename(current_time+".jpg"))
        fa_dst_path = os.path.join(ALIGNMENT_DIR_PATH, current_time+'.bin')
        fp_dst_path = os.path.join(PARSING_DIR_PATH, current_time+'.png')
        # Save Image in file_path
        f.save(img_path)
        
        # face_alignment
        execute_face_alignment(img_path, fa_dst_path)
        evaluate(fp_dst_path, img_path, WEIGHTS_PATH)

        return fa_dst_path

if __name__ == '__main__':
    app.run(debug=True)