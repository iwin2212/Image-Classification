from flask import Flask, flash, request, redirect, url_for, render_template, session, send_from_directory
from werkzeug.utils import secure_filename
import yaml
import os
from const import *
from train_test import train_test
from function import info

UPLOAD_FOLDER = test_path
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file[]")
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                label, mean, std, img_name = train_test()
                os.remove(test_path + filename)
        return render_template('result.html', label=label, acc=mean, std=std, img_name=img_name)
    return render_template('home.html', dict=dict)


if __name__ == '__main__':
    dict = info(train_path)
    app.run(host="127.0.0.1", port=int("80"), debug=True)
