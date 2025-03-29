from flask import Flask, render_template, request, jsonify
from model import run_model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run-model", methods=["POST"])
def run_simulation():
    data = request.json

    try:
        results = run_model(data)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
