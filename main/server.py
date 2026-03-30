from flask import Flask, request
import os

app = Flask(__name__)

SAVE_FOLDER = "fotos"
os.makedirs(SAVE_FOLDER, exist_ok=True)

counter = 0

@app.route('/upload', methods=['POST'])
def upload():
    global counter

    data = request.data

    filename = f"{SAVE_FOLDER}/photo_{counter}.jpg"
    counter += 1

    with open(filename, "wb") as f:
        f.write(data)

    print(f"📸 Guardada {filename}")
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)