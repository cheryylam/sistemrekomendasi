import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')


hotel = pd.read_csv("nusatripnew.csv", header=0)
#preprocessing


def clean_lower(lwr):
    lwr = lwr.lower() # lowercase text
    return lwr

# Buat kolom tambahan untuk data description yang telah dicasefolding  
hotel['lwr1'] = hotel['deskripsi'].apply(clean_lower)
hotel['lwr2'] = hotel['facility'].apply(clean_lower)
hotel['lwr3'] = hotel['nearbyattraction'].apply(clean_lower)
#Remove Puncutuation
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z]')


def remove_punct(text):
    text = clean_spcl.sub('', text)
    text = clean_symbol.sub(' ', text)
    return text

# Buat kolom tambahan untuk data description yang telah diremovepunctuation   
hotel['remove_punct1'] = hotel['lwr1'].apply(remove_punct)
hotel['remove_punct2'] = hotel['lwr2'].apply(remove_punct)
hotel['remove_punct3'] = hotel['lwr3'].apply(remove_punct)
def _normalize_whitespace(text):
    """
    This function normalizes whitespaces, removing duplicates.
    """
    corrected = str(text)
    corrected = re.sub(r"//t",r"\t", corrected)
    corrected = re.sub(r"( )\1+",r"\1", corrected)
    corrected = re.sub(r"(\n)\1+",r"\1", corrected)
    corrected = re.sub(r"(\r)\1+",r"\1", corrected)
    corrected = re.sub(r"(\t)\1+",r"\1", corrected)
    return corrected.strip(" ")
hotel['remove_double_ws1'] = hotel['remove_punct1'].apply(_normalize_whitespace)
hotel['remove_double_ws2'] = hotel['remove_punct2'].apply(_normalize_whitespace)
hotel['remove_double_ws3'] = hotel['remove_punct3'].apply(_normalize_whitespace)
#clean stopwords
stw = open("sw.txt")
# Use this to read file content as a stream:
line = stw.read()
stopword = line.split()

def clean_stopwords(text):
    text = ' '.join(word for word in text.split() if word not in stopword) # hapus stopword dari kolom deskripsi
    return text

# Buat kolom tambahan untuk data description yang telah distopwordsremoval   
hotel['remove_sw1'] = hotel['remove_double_ws1'].apply(clean_stopwords)
hotel['remove_sw2'] = hotel['remove_double_ws2'].apply(clean_stopwords)
hotel['remove_sw3'] = hotel['remove_double_ws3'].apply(clean_stopwords)

wn= nltk.WordNetLemmatizer()
def lemmatization(text):
    text = ' '.join(wn.lemmatize(word) for word in text.split() if word in text)
    return text

# Buat kolom tambahan untuk data description yang telah dilemmatization   
hotel['desc_remove_lemma1'] = hotel['remove_sw1'].apply(lemmatization)
hotel['desc_remove_lemma2'] = hotel['remove_sw2'].apply(lemmatization)
hotel['desc_remove_lemma3'] = hotel['remove_sw3'].apply(lemmatization)
hotel['desc_removefix1']=hotel['desc_remove_lemma1']
hotel['desc_removefix2']=hotel['desc_remove_lemma2']
hotel['desc_removefix3']=hotel['desc_remove_lemma3']
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer= CountVectorizer(encoding='latin-1', ngram_range=(1,1), 
                                  tokenizer=None, analyzer='word',
                                  stop_words= None)
countvec1= count_vectorizer.fit_transform(hotel['desc_removefix1']).toarray()
countvec2= count_vectorizer.fit_transform(hotel['desc_removefix2']).toarray()
countvec3= count_vectorizer.fit_transform(hotel['desc_removefix3']).toarray()
#TF IDF
from sklearn.feature_extraction.text import TfidfTransformer
transformer= TfidfTransformer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False)
tfidf1= transformer.fit_transform(countvec1)
tfidf2= transformer.fit_transform(countvec2)
tfidf3= transformer.fit_transform(countvec3)
#cosine
cos_sim1= cosine_similarity(tfidf1, tfidf1)
cos_sim2= cosine_similarity(tfidf2, tfidf2)
cos_sim3= cosine_similarity(tfidf3, tfidf3)
#recommendation
# Set index utama di kolom 'namahotel'
hotel.set_index('namahotel', inplace=True)
indices = pd.Series(hotel.index)
my_array=indices.to_numpy()
cheryls=my_array

def recommendations1(namahotel, cos_sim1 = cos_sim1):
    
    recommended_hotel = []
    
    # Mengambil nama hotel berdasarkan variabel indicies
    idx = indices[indices == namahotel].index[0]

    # Membuat series berdasarkan skor kesamaan
    score_series = pd.Series(cos_sim1[idx]).sort_values(ascending = False)

    # mengambil index dan dibuat 10 baris rekomendasi terbaik
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    for i in top_10_indexes:
        recommended_hotel.append(list(hotel.index)[i])
        
    return recommended_hotel

def recommendations2(namahotel, cos_sim2 = cos_sim2):
    
    recommended_hotel = []
    
    # Mengambil nama hotel berdasarkan variabel indicies
    idx = indices[indices == namahotel].index[0]

    # Membuat series berdasarkan skor kesamaan
    score_series = pd.Series(cos_sim2[idx]).sort_values(ascending = False)

    # mengambil index dan dibuat 10 baris rekomendasi terbaik
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    for i in top_10_indexes:
        recommended_hotel.append(list(hotel.index)[i])
        
    return recommended_hotel

def recommendations3(namahotel, cos_sim3 = cos_sim3):
    
    recommended_hotel = []
    
    # Mengambil nama hotel berdasarkan variabel indicies
    idx = indices[indices == namahotel].index[0]

    # Membuat series berdasarkan skor kesamaan
    score_series = pd.Series(cos_sim3[idx]).sort_values(ascending = False)

    # mengambil index dan dibuat 10 baris rekomendasi terbaik
    top_10_indexes = list(score_series.iloc[1:15].index)
    top_10_cosim = list(score_series.iloc[1:15])
    for i, cosim in zip(top_10_indexes, top_10_cosim):
        if cosim == 1 or cosim == 0:
            pass
        else :
            recommended_hotel.append(list(hotel.index)[i])
    return recommended_hotel

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('home.html', cheryls=cheryls)

    if request.method == 'POST':
        hotels = request.form['daftarhotel']
        res1 = recommendations1(hotels)
        res2 = recommendations2(hotels)
        res3 = recommendations3(hotels)
        return render_template('akhirfix.html', result1=res1, result2=res2, result3=res3, cheryls=cheryls)
    else:
        return render_template('home.html')
 

if __name__ == '__main__':
    app.run(debug=True)
