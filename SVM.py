# import torch
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import torchvision
import os
os.environ["OMP_NUM_THREADS"] = "1"
def cm_result(cm):
    # Calculate the accuracy of a confusion_matrix, where parameter 'cm' means confusion_matrix.
    a = cm.shape
    corrPred = 0
    falsePred = 0
    
    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred += cm[row, c]
            else:
                falsePred += cm[row, c]
    Accuracy = corrPred / (cm.sum())
    return Accuracy

if __name__ == '__main__':
    # Importing the dataset using torchvision
    train_data = torchvision.datasets.MNIST('./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST('./mnist', train=False, transform=torchvision.transforms.ToTensor())

    # Extracting features (X) and labels (y) from the data
    X_train = train_data.data.view(-1, 28 * 28).numpy()
    y_train = train_data.targets.numpy()
    X_test = test_data.data.view(-1, 28 * 28).numpy()
    y_test = test_data.targets.numpy()

    # Splitting the dataset into the Training set and Test set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=40)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    # for g in [0.001,0.01, 0.1, 0.25, 1, 10]:
    # for g in [1/784,0.001,0.01,0.1,1,10]:
    for g in [1/784]:
        # print('g=', g)
        # for c in [0.1,1,10,100]:
        for c in [100]:
    # g=0.03
    # c=100
            print('g=',g, 'c=',c)

            start = time.time()
            # Fitting SVC Classification to the Training set with RBF kernel
            svcclassifier = SVC(kernel='rbf', random_state=0, C=c, gamma=g)
            svcclassifier.fit(X_train, y_train)

            # Predicting the Validation set results
            y_pred_val = svcclassifier.predict(X_val)

            # Making the Confusion Matrix for Validation set
            cm_val = confusion_matrix(y_val, y_pred_val)
            print("Confusion Matrix for Validation Set:")
            print(cm_val)

            ValAccuracy = cm_result(cm_val)
            print('Accuracy on Validation Set is: ', round(ValAccuracy * 100, 2))

            # Predicting the Test set results
            y_pred_test = svcclassifier.predict(X_test)

            # Making the Confusion Matrix for Test set
            cm_test = confusion_matrix(y_test, y_pred_test)
            print("\nConfusion Matrix for Test Set:")
            print(cm_test)

            TestAccuracy = cm_result(cm_test)
            print('Accuracy on Test Set is: ', round(TestAccuracy * 100, 2))
            end = time.time()
            print('Done in {}s'.format(end - start))    
