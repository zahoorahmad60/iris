from flask import Flask, render_template,request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    feature_values = [float(x) for x in request.form.values()]
    array = [np.array(feature_values)]
    predict = model.predict(array)

    return render_template("index.html", prediction_text=f"The Flower is {predict} ")

if __name__ == "__main__":
    app.run(debug=True)