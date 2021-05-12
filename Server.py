import os
from flask import Flask, flash, request, redirect, url_for
import fencingEngine


app = Flask(__name__)


@app.route("/")
def home():
    return "<h1>hello world</h1>"


@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['video']
        f.save('video.mp4')
        technique = fencingEngine.get_technique('video.mp4')
        os.remove('video.mp4')
        print(technique)

        return technique


app.run()