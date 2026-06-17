from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    news_text = ""
    confidence = ""

    if request.method == "POST" and request.form["news"]:
        news_text = request.form["news"]
        result = model.predict([news_text])[0]
        prob = model.predict_proba([news_text])[0]
        confidence = round(max(prob) * 100, 1)
        prediction = result.upper()

    return render_template("index.html",
        prediction=prediction,
        news_text=news_text,
        confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)