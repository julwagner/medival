from flask import Flask, render_template, request
from logic import evaluate_uploaded_file

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    file = request.files["pdf"]
    result = evaluate_uploaded_file(file, app.config["UPLOAD_FOLDER"])
    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
