import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from joblib import dump, load


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# encode labels. e.g. "Female" is "1, 0" and "Male" is "0, 1"
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# one-hot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
dump(sc, 'std_scaler.bin', compress=True)


## Part 2 - Building the ANN

### Initializing the ANN

ann = tf.keras.models.Sequential()

### Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

### Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

### Adding the output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

## Part 3 - Training the ANN

### Compiling the ANN

# when category loss must be "category_crossentropy"
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Training the ANN on the Training set

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# serialize model to JSON
model_json = ann.to_json()
with open("config.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
ann.save_weights("model.h5")
print("Saved model to disk")
 