import argparse
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import warnings
import json
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')


def read_json_data():
    path = 'yummly.json' 
    if os.path.exists(path) :
        df = pd.read_json(path)
        ingredient_list = []
        cuisine_list = []
        id_list = []
# creating lists for all fields in the json file.
        for lst_no in range(len(df)):
            str_lst = ' '.join(df.iloc[lst_no,2])
            ingredient_list.append(str_lst)
            cuisine_list.append(df.iloc[lst_no,1])
            id_list.append(df.iloc[lst_no,0])

        return ingredient_list, cuisine_list, id_list
    else:
        sys.stderr.write("No yummly.json")

# vectorizing text to features
def vectorize_corpus(recepie_lst,ingredients):
    
    ingredients = str(ingredients)
    # apppending user input to the end
    recepie_lst.append(ingredients)
    vectorizer = TfidfVectorizer(use_idf = True, stop_words = 'english')
    vector = vectorizer.fit_transform(recepie_lst)
    vector = vector.todense()
    return vector

def train_test(data):
    # the last row which is the user input is set aside for testing
    train = data[:-1]
    test = data[-1]
    return train,test

def train_model(train,cuisine,N):
   # The KNN model is trained on the data 
   n_foods = KNeighborsClassifier(N)
   knn = n_foods.fit(train,cuisine)
   return knn


def n_closest_cuisines(test,n_foods,N):
    # the cuisine and it's N closest neighbours are predicted
    prediction = n_foods.predict(test)
    score = n_foods.predict_proba(test)[0].max()
    return prediction, score


 
if __name__== "__main__":

    arguments = argparse.ArgumentParser()

    arguments.add_argument("--N",type = int, required= True, help = "N closest foods")
    arguments.add_argument("--ingredients",type = str ,required = True, help = "Enter the ingredients that make up the Cuisine", nargs = "*", action = "append")
    
    args = arguments.parse_args()
    ingredient_list, cuisine_list, id_list = read_json_data()
    data = vectorize_corpus(ingredient_list,args.ingredients)
    train,test = train_test(data)
    n_foods  = train_model(train,cuisine_list,args.N)
    prediction, score = n_closest_cuisines(test,n_foods,args.N)
    neighbours = n_foods.kneighbors(test)
    
    # outputing to stdout in json format.
    store = dict([("cuisine",str(prediction[0])),("score",round(score,2)),("closest",[])])
    scores =  neighbours[0].flatten()
    foods = neighbours[1].flatten()
    for i in range(args.N):
       X =  cosine_similarity(data[foods[i]], data[-1])[0][0]
       store["closest"].append(dict([("id",str(id_list[foods[i]])),("score",round(X,2))]))
    jstr = json.dumps(store, indent=4)
    sys.stdout.write(jstr)   
