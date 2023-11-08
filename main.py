from flask import Flask, render_template, request, flash, url_for
from werkzeug.utils import secure_filename
from processimage import predict
import os
import cv2

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect" , methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return "error no file selected"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new = predict(filename)
            # new2 = predict2(filename)

            flash(f"Find the processed image <a href = '{new}' target= '_blank'>here</a>")

            # flash(f"Find the processed image <a href = '{new2}' target= '_blank'>here</a>")
            return render_template("index.html")
    return render_template("index.html")

app.run(debug=True)