import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def process(img, kern):
    accum = np.zeros_like(img)
    return cv2.filter2D(img, cv2.CV_8UC3, kern)

trainDataRaw = np.genfromtxt('/home/ben/Data/kaggle/digitRecogniser/train.csv', skip_header=1, delimiter=",")
testData = np.genfromtxt('/home/ben/Data/kaggle/digitRecogniser/test.csv', skip_header=1, delimiter=",")
trainLabels = trainDataRaw[:100,0].astype(str)
trainData = trainDataRaw[:100,1:]
del trainDataRaw

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('clf', SVC(kernel = 'rbf', C = 1, gamma = 'auto'))
    ])
scoresRaw = np.mean(cross_val_score(pipe, trainData, trainLabels, cv=3, n_jobs=-1))


filters = []
ksize = 15
for theta in np.arange(0, np.pi, np.pi / 8):
    for sigma in [2, 3, 4, 5]:
        for lamba in [10]:
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamba, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)

trainingGaborActivations = []
trainImages = trainData.reshape([-1,28,28])
del trainData
for i, img in enumerate(trainImages):
    trainingGaborActivations.append([])

    for fil in filters:
        filteredImage = process(img, fil)
        trainingGaborActivations[i].append(np.mean(np.array(filteredImage)))

scoresGabor = np.mean(cross_val_score(pipe, np.array(trainingGaborActivations), trainLabels, cv=3, n_jobs=-1))

print('Average raw testing accuracy: {}, chance: {}'.format(scoresRaw, 0.1))
print('Average Gabor testing accuracy: {}, chance: {}'.format(scoresGabor, 0.1))

testingGaborActivations = []
testImages = testData.reshape([-1,28,28])
for i, img in enumerate(testImages):
    testingGaborActivations.append([])

    for fil in filters:
        filteredImage = process(img, fil)
        testingGaborActivations[i].append(np.mean(np.array(filteredImage)))

pipe.fit(np.array(trainingGaborActivations), trainLabels)
predictedLabels = pipe.predict(np.array(testingGaborActivations))