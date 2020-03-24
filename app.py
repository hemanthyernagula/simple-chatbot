from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


app = Flask(__name__)

data = pd.read_csv('data/data.csv')
tfidf = TfidfVectorizer()
tfidf.fit_transform(data.questions.values)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    query = tfidf.transform([userText])

    try:
        answer= data.answers.iloc[np.max([j for j,k in enumerate([cosine_similarity(query,tfidf.transform([data.questions.iloc[i]]))[0][0] for i 					in range(data.shape[0])]) if k>0.6 ])]

        print(answer)
        return str(answer)
    except:

        return 'Sorry I did not understand that!!'


if __name__ == "__main__":
    app.run()
