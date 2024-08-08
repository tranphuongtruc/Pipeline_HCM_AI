from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import json

from utils.query_processing import Translation
from utils.faiss import Myfaiss

app = Flask(__name__, template_folder='templates')

####### CONFIG #########
with open('image_path.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {int(key): value for key, value in json_dict.items()}
LenDictPath = len(DictImagePath)
bin_file = 'faiss_normal_ViT.bin'
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")
########################

@app.route('/home')
@app.route('/')
def thumbnailimg():
    index = int(request.args.get('index', 0))
    imgperindex = 100

    first_index = index * imgperindex
    last_index = min(first_index + imgperindex, LenDictPath)

    page_filelist = [DictImagePath[i] for i in range(first_index, last_index)]
    pagefile = [{'imgpath': path, 'id': i} for i, path in enumerate(page_filelist, start=first_index)]

    data = {'num_page': (LenDictPath + imgperindex - 1) // imgperindex, 'pagefile': pagefile}

    return render_template('home.html', data=data)

@app.route('/imgsearch')
def image_search():
    id_query = int(request.args.get('imgid'))
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=50)

    pagefile = [{'imgpath': imgpath, 'id': int(id)} for imgpath, id in zip(list_image_paths, list_ids)]
    data = {'num_page': 1, 'pagefile': pagefile}

    return render_template('home.html', data=data)

@app.route('/textsearch')
def text_search():
    text_query = request.args.get('textquery')
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=50)

    pagefile = [{'imgpath': imgpath, 'id': int(id)} for imgpath, id in zip(list_image_paths, list_ids)]
    data = {'num_page': 1, 'pagefile': pagefile}

    return render_template('home.html', data=data)

@app.route('/get_img')
def get_img():
    fpath = request.args.get('fpath')
    image_name = "/".join(fpath.split("/")[-2:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280, 720))
    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
