# importing the modules

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# loading the data
text = input('input csv file name:')


df = pd.read_csv(text)

data = df['news']

# Using the model
with open('tfidfmodel.pickle','rb') as f:
    tf_idf_vectorizor = pickle.load(f)
    
    
tf_idf = tf_idf_vectorizor.fit_transform(data)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()

with open('model.pickle','rb') as f:
    model = pickle.load(f)
    
prediction = model.predict(tf_idf_array)

# printing the clusters
print(np.unique(prediction))
print(prediction[0:1000])       
