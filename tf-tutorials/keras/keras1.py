from keras.models import Sequential 
from keras.layers import Dense
import numpy as np 

#Attribute Information:

#1. Number of times pregnant
#2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#3. Diastolic blood pressure (mm Hg)
#4. Triceps skin fold thickness (mm)
#5. 2-Hour serum insulin (mu U/ml)
#6. Body mass index (weight in kg/(height in m)^2)
#7. Diabetes pedigree function
#8. Age (years)
#9. Class variable (0 or 1) 

seed=7
np.random.seed(seed)

dataset=np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

print dataset.shape

X=dataset[:,0:8]
Y=dataset[:,8]

model=Sequential()

model.add(Dense(20,input_dim=8,init='uniform',activation='relu'))
model.add(Dense(12,init="uniform",activation="relu"))
model.add(Dense(1,init="uniform",activation='sigmoid')) # used to preditct 1/0, softmax for many classes

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

model.fit(x=X, y=Y, batch_size=100, nb_epoch=200)

loss, accuracy = model.evaluate(X, Y)

print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5. make predictions
probabilities = model.predict(X)
predictions = [float(round(x)) for x in probabilities]
accuracy = np.mean(predictions == Y)

print("Predicted Accuracy %.2ff%%" % (accuracy*100))








