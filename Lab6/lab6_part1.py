import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#load data
mat = scipy.io.loadmat('2Class_KNN_Data.mat')

#separate matrix data
C1 = mat['c1']
C2 = mat['c2']
C1_test = mat['c1test']
C2_test = mat['c2test']

#convert data as a np array
C1 = np.array(C1)
C2 = np.array(C2)
C1_test = np.array(C1_test)
C2_test = np.array(C2_test)

#plot the scatter plot of the classes and their tests
plt.scatter(C1[:,0], C1[:,1], c='blue', label="C1")
plt.scatter(C2[:,0], C2[:,1], c = 'red', label='C2')
plt.scatter(C2_test[:,0], C2_test[:,1], c = 'green', label='C2 test')
plt.scatter(C1_test[:,0], C1_test[:,1], c = 'yellow', label='C1 test')
plt.legend()
#plt.show()

#Create labels for C1 and C2
C1_labels = np.zeros((100, 1))
C2_labels = np.ones((100, 1))


labels = np.concatenate((C1_labels, C2_labels))
train_data = np.concatenate((C1, C2))

#use k-nn
knn_clf = KNeighborsClassifier()
knn_clf.fit(train_data, labels)

#use test data
C1_test_labels = np.zeros((20, 1))
C2_test_labels = np.ones((20, 1))
test_labels = np.concatenate((C1_test_labels, C2_test_labels))

data_test = np.concatenate((C1_test, C2_test))
prediction = knn_clf.predict(data_test)

#get classification error
print("Classification error")
print(accuracy_score(test_labels, prediction))

#print confusion matrix
print("Confusion matrix")
print(confusion_matrix(test_labels, prediction))



