from flask import Flask, flash, request, redirect, url_for, render_template, session
from function import data, addData, delDevice, specificList, editDevice
import yaml
import os

app = Flask(__name__)
app.secret_key = 'any random'


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug="true")
