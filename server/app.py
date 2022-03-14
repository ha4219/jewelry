from this import d
from flask import Flask, redirect, render_template, request, url_for, send_from_directory, abort
from graphviz import render
from werkzeug.utils import secure_filename
from werkzeug.exceptions import NotFound
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
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "coin"))
from coinWrapper import coin_generator
from skimage import io
import numpy as np
from PIL import Image
import ctypes
import json
app = Flask(__name__)

# SET PATH
IMG_DIR_PATH = os.path.join('assets', 'images')
ALIGNMENT_DIR_PATH = os.path.join('assets', 'face_alignment_results')
PARSING_DIR_PATH = os.path.join('assets', 'face_parsing_results')
COIN_DIR_PATH = os.path.join('assets', 'coin_generator_results')
BACK_IMAGE_DIR_PATH = os.path.join('assets', 'back_images')
TEXT_IMAGE_DIR_PATH = os.path.join('assets', 'text_images')
WEIGHTS_PATH = "../Face-parsing/res/cp/79999_iter.pth"
COIN_LIB_PATH="../coin/lib.so"

# Model for Face Alignment
FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

# Model for Face Parsing
NET = BiSeNet(19)
NET.cuda()
NET.load_state_dict(torch.load(WEIGHTS_PATH))
NET.eval()

# load lib.so
lib = ctypes.cdll.LoadLibrary(COIN_LIB_PATH)

def image_save_with_resize(image, image_path):
    img = Image.open(image)
    img = img.resize((512, 512), Image.BILINEAR)
    img.convert("RGB").save(image_path)

def image_save_without_resize(image, image_path):
    img = Image.open(image)
    img.convert("RGB").save(image, image_path)

def execute_face_alignment(img_path, dst_path):
    input_ = io.imread(img_path)
    preds = FA.get_landmarks(input_)
    np.array(preds).tofile(dst_path)

@app.route('/assets/images/<path>')
def image(path):
    try:
        return send_from_directory(IMG_DIR_PATH, path=path)
    except FileNotFoundError:
        abort(404)

@app.route('/assets/face_alignment_results/<path>')
def image2(path):
    try:
        return send_from_directory(ALIGNMENT_DIR_PATH, path=path)
    except FileNotFoundError:
        abort(404)

@app.route('/assets/face_parsing_results/<path>')
def image3(path):
    try:
        return send_from_directory(PARSING_DIR_PATH, path=path)
    except FileNotFoundError:
        abort(404)

@app.route('/assets/coin_generator_results/<path>')
def image4(path):
    try:
        return send_from_directory(COIN_DIR_PATH, path=path)
    except FileNotFoundError:
        abort(404)

@app.route('/face_alignment', methods=['POST'])
def post_face_alignment(): 
    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")

    if request.files:
        f = request.files['front']
        image_path = os.path.join(IMG_DIR_PATH, secure_filename(current_time+"."+f.filename.split('.')[-1]))
        image_save_with_resize(f, image_path)
    else:
        image_path = os.path.join(IMG_DIR_PATH,request.form.get('front'))
        if not os.path.isfile(image_path):
            raise NotFound('Image file not found')
            
    fa_dst_path = os.path.join(ALIGNMENT_DIR_PATH, current_time+'.bin')
    
    execute_face_alignment(image_path, fa_dst_path)

    dc = {'image_path' : image_path, 'fa_dst_path' : fa_dst_path}
    return json.dumps(dc)

@app.route('/face_parsing', methods=['POST'])
def post_face_parsing():

    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")

    if request.files:
        f = request.files['front']
        image_path = os.path.join(IMG_DIR_PATH, secure_filename(current_time+"."+f.filename.split('.')[-1]))
        image_save_with_resize(f, image_path)
    else:
        image_path = os.path.join(IMG_DIR_PATH,request.form.get('front'))
        if not os.path.isfile(image_path):
            raise NotFound('Image file not found')
    
    fp_dst_path = os.path.join(PARSING_DIR_PATH, current_time+'.png')
    
    execute_face_parsing(image_path, fp_dst_path, NET)
    
    dc = {'image_path' : image_path, 'fp_dst_path' : fp_dst_path}
    return json.dumps(dc)

# for test Using front, text, back, not alignment and parsing
@app.route('/coin_generating', methods=['POST'])
def post_coin_generating():
    req = request.files
    req_form = request.form
    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")

    file_list = []
    for i in req:
        file_list.append(i)

    # file or text
    if 'front' in file_list and req['front'].filename != '':
        f = req['front']
        front_image_path = os.path.join(IMG_DIR_PATH, secure_filename(current_time+"."+f.filename.split('.')[-1]))
        image_save_with_resize(f, front_image_path)
    elif req_form.get('front'):
        front_image_path = os.path.join(IMG_DIR_PATH,req_form.get('front'))
        if not os.path.isfile(front_image_path):
            raise NotFound('Front image file not found')
    else:
        front_image_path = '../coin/007F.png'

    if 'text' in file_list and req['text'].filename != '':
        f = req['text']
        text_image_path = os.path.join(TEXT_IMAGE_DIR_PATH, secure_filename(current_time+"."+f.filename.split('.')[-1]))
        image_save_with_resize(f, text_image_path)
    elif req_form.get('text'):
        text_image_path = os.path.join(TEXT_IMAGE_DIR_PATH,request.form.get('text'))
        if not os.path.isfile(text_image_path):
            raise NotFound('Text Image file not found')
    else:
        text_image_path = '../coin/007F_TEXT.png'

    if 'back' in file_list and req['back'].filename != '':
        f = req['back']
        back_image_path = os.path.join(BACK_IMAGE_DIR_PATH, secure_filename(current_time+"."+f.filename.split('.')[-1]))
        image_save_with_resize(f, back_image_path)
    elif req_form.get('back'):
        back_image_path = os.path.join(BACK_IMG_DIR_PATH,request.form.get('back'))
        if not os.path.isfile(back_image_path):
            raise NotFound('Back Image file not found')
    else:
        back_image_path = '../coin/003R.png'

    if 'face_alignment' in file_list and req['face_alignment'].filename != '':
        f = req['face_alignment']
        face_alignment_path = os.path.join(ALIGNMENT_DIR_PATH, secure_filename(current_time+"."+f.filename.split('.')[-1]))
    elif req_form.get('face_alignment'):
        face_alignment_path = os.path.join(ALIGNMENT_DIR_PATH,request.form.get('face_alignment'))
        if not os.path.isfile(face_alignment_path):
            raise NotFound('Face_alignment file not found')
    else:
        face_alignment_path = "NONE"

    if 'face_parsing' in file_list and req['face_parsing'].filename != '':
        f = req['face_parsing']
        face_parsing_path = os.path.join(PARSING_DIR_PATH, secure_filename(current_time+"."+f.filename.split('.')[-1]))
    elif req_form.get('face_parsing'):
        face_parsing_path = os.path.join(PARSING_DIR_PATH,request.form.get('face_parsing'))
        if not os.path.isfile(face_parsing_path):
            raise NotFound('Face_parsing file not found')
    else:
        face_parsing_path = "NONE"


    coin_outp_path = os.path.join(COIN_DIR_PATH, current_time+'.stl')
     
    arg = [
        "",
        front_image_path,
        face_parsing_path,
        face_alignment_path,
        text_image_path,
        coin_outp_path,
        back_image_path,
    ]
    coin_generator(lib, arg)
    
    dc = {
            'front_image_path' : front_image_path,
            'text_image_path' : text_image_path,
            'back_image_path' : back_image_path,
            'face_alignment_path' : face_alignment_path,
            'face_parsing_path' : face_parsing_path,
            'coin_outp_path' : coin_outp_path
         }
    return json.dumps(dc)
    
@app.route('/uploader', methods=['POST'])
def uploader_file():
    req = request.files
    
    current_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
    front_image_path = os.path.join(IMG_DIR_PATH, secure_filename(current_time+"."+req['front'].filename.split('.')[-1]))
    text_image_path = os.path.join(TEXT_IMAGE_DIR_PATH, secure_filename(current_time+"."+req['text'].filename.split('.')[-1]))
    back_image_path = os.path.join(BACK_IMAGE_DIR_PATH, secure_filename(current_time+"."+req['back'].filename.split('.')[-1]))
    coin_dst_path = os.path.join(COIN_DIR_PATH, current_time+'.stl')
    fa_dst_path = os.path.join(ALIGNMENT_DIR_PATH, current_time+'.bin')
    fp_dst_path = os.path.join(PARSING_DIR_PATH, current_time+'.png')
    # Save Image in file_path
    image_save_with_resize(req['front'], front_image_path) 
    if req['text']:
        image_save_without_resize(req['text'], text_image_path)
    if req['back']:
        image_save_with_resize(req['back'], back_image_path)

    
    # face_alignment
    execute_face_alignment(front_image_path, fa_dst_path)
    # face_parsing
    execute_face_parsing(front_image_path, fp_dst_path, NET)

    # test coin generator, arg : input
    arg = [
        "",
        front_image_path if req['front'] else "../coin/007F.png",
        "NONE",
        "NONE",
        text_image_path if req['text'] else "../coin/007F_TEXT.png",
        coin_dst_path if req['front'] else "../coin/tmp.stl",
        back_image_path if req['back'] else "../coin/003R.png",
    ]
    
    coin_generator(lib, arg)
    
    dc = {
            'front_image_path' : front_image_path,
            'fa_dst_path' : fa_dst_path,
            'fp_dst_path' : fp_dst_path,
            'coin_dst_path' : coin_dst_path
        }
    return json.dumps(dc)
@app.route('/test', methods=['POST'])
def test():
    req_form = request.files
    return '1' if req_form['frtr'] else '0'

if __name__ == '__main__':
    app.run(debug=True)
