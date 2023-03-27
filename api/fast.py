# $DELETE_BEGIN
import pytz
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import spacy
import en_core_web_sm
import api.preprocessing as preproc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#load verctorizer and model
vectorizer = joblib.load("api/tfidf_vectorizer.pkl", 'r')
multilabel_binarizer = joblib.load("api/multilabel_binarizer.pkl", 'r')
model = joblib.load("api/logit_model.pkl", 'r')


# Define the default route
@app.get("/")
def root():
    return {"message": "Welcome to Your StackOverflow Question Classification FastAPI"}


@app.get("/predict")
def predict(question):

    # Clean the question
    nlp = spacy.load("en_core_web_sm")
    #nlp = spacy.load('en_core_web_md', exclude=['tok2vec', 'ner', 'parser', 'attribute_ruler', 'lemmatizer'])
    pos_list = ["NOUN","PROPN"]
    rawdoc = question
    cleaned_question = preproc.text_cleaner(rawdoc, nlp, pos_list, "english")

    # Apply TfidfVectorizer
    X_tfidf = vectorizer.transform([cleaned_question])

    # Perform prediction
    predict = model.predict(X_tfidf)
    predict_probas = model.predict_proba(X_tfidf)
    # Inverse multilabel binarizer
    tags_predict = multilabel_binarizer.inverse_transform(predict)

    # DataFrame of probas
    df_predict_probas = pd.DataFrame(columns=['Tags', 'Probas'])
    df_predict_probas['Tags'] = multilabel_binarizer.classes_
    df_predict_probas['Probas'] = predict_probas.reshape(-1)
    # Select probas > 33%
    df_predict_probas = df_predict_probas[df_predict_probas['Probas']>=0.33]\
        .sort_values('Probas', ascending=False)

    # Results
    results = {}
    results['Predicted_Tags'] = tags_predict
    results['Predicted_Tags_Probabilities'] = df_predict_probas\
        .set_index('Tags')['Probas'].to_dict()

    return results, 200


# $DELETE_END
