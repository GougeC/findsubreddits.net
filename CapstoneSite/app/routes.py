import sys
sys.path.append("..") # Adds higher directory to python modules path.

from flask import render_template,request, flash, redirect,url_for,jsonify
from app import app
import reddit_recommenders as rrec

#modelpath = '../data/1Convbigdense.HDF5'
#smpath = '../data/1Convbigdensedict.pkl'
#wipath = '../data/1Convbigdenseindex1.pkl'
#conv1 = rrec.CNN_reddit_recommender(modelpath,smpath,wipath)


@app.route('/')
def index():
    return render_template('index.html',title = 'Home')

@app.route('/twitter', methods=['POST'])
def get_twitter_recs():
    user_submission = request.json
    print(request.json)
    twitter_accs = user_submission['handles']
    preds = conv1.predict_on_list_handles(twitter_accs,10)
    print(preds)
    return jsonify(preds)

@app.route('/text', methods=['POST'])
def get_text_recs():
    user_submission = request.json
    print(request.json)
    text= user_submission['text']
    preds = conv1.predict_on_text(text,10)
    print(preds)
    return jsonify(preds)
