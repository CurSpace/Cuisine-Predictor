# cs5293sp22-project2

### Author : Pradipkumar Rajasekaran

__Summary:__

The goal of this project is to take a list of ingredients form the user and predict the type of cuisine that the provided ingredients make 
and the N meals similar to the prediction. The N will be provided by the user as a command line argument and the ingredients as well. 

## Developement Process

- Created project2.py with function to train and test KNN using the given yummly.json file.
- Read the json data as a dataframe using pd.read_json(path)
- The data is split into lists and then vectorized.
- Then the data is split into test and training set. The user input is appended to the end to be used as testing data.
- The training data is fit to a KNN model to find N clusters.
- Predictions are made scores are assigned to the predicted cuisine and the neighbouring N cusines.

__Installation__


1. Clone the repository- git clone https://github.com/CurSpace/cs5293sp22-project2.git
2. Navigate into the repo folder(cs5293sp22-project2)
3. Install all requirements using - pipenv install
4. From the repo folder run - pipenv run python -m pytest


__Python packages used:__

- sklearn
- json
- nltk 
- argparser
- os
- pytest
- sys
- pandas
- warnings

### Run Code

- Run the program by:
```
   pipenv run python project2.py --N [no of neighbours] --ingredients [ingredients that make up the cuisine]
```

   Ex: 
   
   ```
       pipenv run python project2.py --N 5 --ingredient "vegetable oil" --ingredient "wheat" --ingredient salt --ingredient garlic --ingredient pepper
   ```
   __Sample Output:__
   
 ```
{
    "cuisine": "indian",
    "score": 1.0,
    "closest": [
        {
            "id": "22213",
            "score": 0.87
        },
        {
            "id": "5366",
            "score": 0.75
        },
        {
            "id": "22463",
            "score": 0.66
        },
        {
            "id": "28617",
            "score": 0.63
        },
        {
            "id": "11488",
            "score": 0.6
        }
    ]
}
 ```
 
 __Description of User Defined Functions:__
 
 1. read_json_data() - Reads the json file and stores the data into a data frame.
                     - Then, the data frame is broken into lists(ingredient_list, cuisine_list, id_list).
                     - And returns the lists.

 2. vectorize_corpus(recepie_lst,ingredients)
                     - Appends the user input to the data.
                     - Then vectorizes the data.
                     - And returns the vector. 

 3. train_test(data) - The data from the file(yummlu.json) is kept as traning data and the user input is set aside for testing.
                     - Returns train,test.

 4. train_model(train,cuisine,N)
                     - The KNN classifier is trained on the training data.

 5. n_closest_cuisines(test,n_foods,N)
                     - Predictions are made on the test data.
                     - prediction and score are returned.
                                       
 
 
 
__Assumptions__

1. The yummly.json file is present in the same folder as project2.py
  
2. When the user inputs ingredients, they are added to the end of the dataset. The user input is the test data. 

3. Cosine similarity between the the nearest neighbours and the user input cuisine is a good metric of score for the neighbours.

4. Score of predicted cuisine is the probability of that cuisine belonging to the class.
 
__Testing__

1. testing_read_json_data() - Tests if the len of the returned lists but the read_json_data()function is 39774

2. testing_vectorize_corups() - Tests if vectorize_corpus(ingredient_list,ingredients) returns a 2d array. 
 
3. testing_train_test() - Tests if the training data has of shape(39774, 2970) and testing data has shape(1, 2970)

4.testing_train_model() - Tests if train_model(train,cuisine_list,N) returns 'KNeighborsClassifier()'

5. testing_n_closese_cuisine() - Tests if prediction[0] == 'indian' and score == 1.0

