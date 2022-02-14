from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sys
# for import face_alignment
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "Face-alignment"))
import face_alignment
from skimage import io
import numpy as np
app = Flask(__name__)

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

        # assets/~~~.jpg
        file_path = os.path.join('assets', secure_filename(f.filename))
        # assets/face_alignment_results/~~~.bin
        dst_path = os.path.join('assets', 'face_alignment_results', f.filename.split('.')[-2]+'.bin')
        # Save Image in file_path
        f.save(file_path)
        
        # face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
        input_ = io.imread(file_path)
        preds = fa.get_landmarks(input_)
        np.array(preds).tofile(dst_path)

        return dst_path

if __name__ == '__main__':
    app.run(debug=True)