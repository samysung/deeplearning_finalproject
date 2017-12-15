import numpy as np
from sklearn import svm

traindata = np.load('H_patches.npy')
trainlabel = np.load('H_labels_fixed.npy')
testdata = np.load('H_patches_test.npy')

train = np.empty([206,2])
for i in range(206):
    a = np.histogram(traindata[i], bins=[0, 1, 2, 3])
    a_ = a[0]
    train[i,0] = a_[1]
    train[i,1] = a_[2]

    i = i+1

print(train)

test = np.empty([40,2])
for i in range(40):
    b = np.histogram(testdata[i], bins=[0, 1, 2, 3])
    b_ = b[0]
    test[i,0] = b_[1]
    test[i,1] = b_[2]
    i = i+1

print(test)

exit(3)

# Take the first two features. We could avoid this by using a two-dim dataset
X = train
y = trainlabel

C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
models = clf.fit(X, y)
result = clf.predict(test)
print(result)


