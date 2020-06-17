from flask import Flask, flash, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
import yaml
import os
from const import *
from train import train
from test import test
from function import *

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
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        test(models, trainDataGlobal, trainLabelsGlobal, train_labels)
        return render_template('result.html')
    dict = info(train_path)
    models, trainDataGlobal, trainLabelsGlobal, train_labels, result = train()
    return render_template('home.html', dict=dict, rate=result)


if __name__ == '__main__':
    app.run(debug="true")
