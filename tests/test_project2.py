
import project2
import pytest




ingredients = ["vegetable oil","wheat","salt","garlic","pepper"]
N = len(ingredients)
def testing_read_json_data():#data_file):
    ingredient_list, cuisine_list, id_list = project2.read_json_data()#test_file)
    
    assert len(ingredient_list) == 39774

    assert len(cuisine_list) == 39774

    assert len(id_list) == 39774

def testing_vectorize_corups():
    ingredient_list, cuisine_list, id_list = project2.read_json_data()
    data = project2.vectorize_corpus(ingredient_list,ingredients)

    assert len(data.shape) == 2

def testing_train_test():
    ingredient_list, cuisine_list, id_list = project2.read_json_data()
    data = project2.vectorize_corpus(ingredient_list,ingredients)
    train,test =project2.train_test(data)
    assert train.shape == (39774, 2970)
    assert test.shape == (1, 2970)

def testing_train_model():
    ingredient_list, cuisine_list, id_list = project2.read_json_data()
    data = project2.vectorize_corpus(ingredient_list,ingredients)
    train,test =project2.train_test(data)
    n_foods  = project2.train_model(train,cuisine_list,N)
    assert str(n_foods) == 'KNeighborsClassifier()'

def testing_n_closese_cuisine():
    ingredient_list, cuisine_list, id_list = project2.read_json_data()
    data = project2.vectorize_corpus(ingredient_list,ingredients)
    train,test =project2.train_test(data)
    n_foods  = project2.train_model(train,cuisine_list,N)
    prediction, score = project2.n_closest_cuisines(test,n_foods,N)
    assert prediction[0] == 'indian'
    assert score == 1.0
