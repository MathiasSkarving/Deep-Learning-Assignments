import random
from cmath import sqrt

import math
import matplotlib.pyplot as plt
import numpy as np
from linearclassifier import Loss_type
from functools import partial

from medmnist import BloodMNIST

from knn import KNearestNeighbor
from linearclassifier import LinearClassifier

trainDataset = BloodMNIST(split="train", download=True,size=28)
valDataset = BloodMNIST(split="val", download=True,size=28)
testDataset = BloodMNIST(split="test", download=True,size=28)

trainImages,trainLabels,trainInfo = trainDataset.__dict__['imgs'],trainDataset.__dict__['labels'],trainDataset.__dict__['info']['label']


print('Training data:')
print(f'Images: {trainImages.shape}, Labels: {trainLabels.shape}')

valImages,valLabels = valDataset.__dict__['imgs'],valDataset.__dict__['labels']
print('Validation data:')
print(f'Images: {valImages.shape}, Labels: {valLabels.shape}')

testImages,testLabels = testDataset.__dict__['imgs'],testDataset.__dict__['labels']
print('Test data:')
print(f'Images: {testImages.shape}, Labels: {testLabels.shape}')

print('\n')
print('plot some examples:')

random.seed(42)
fig, axes = plt.subplots(5, len(trainInfo), figsize=(15, 5))

for class_,name in trainInfo.items():
    print(f'Class: {class_}, class name: {name}. Number of training samples: {len(trainLabels[trainLabels==int(class_)])}')
    # Get indices of all images belonging to class i
    class_indices = [idx for idx, label in enumerate(trainLabels) if int(class_) == label]
    # Randomly select 5 indices
    selected_indices = random.sample(class_indices, 5)
    for j, idx in enumerate(selected_indices):
        image, label = trainImages[idx],trainLabels[idx]
        axes[j, int(class_)].imshow(image, cmap='gray')
        axes[j, int(class_)].axis('off')
        if j == 0:
            axes[j, int(class_)].set_title(f'{name[:5]}: {class_}')

plt.tight_layout()
plt.show()

#set random seed for reproducibility
np.random.seed(0)

#Shuffle the data
randomize = np.arange(trainImages.shape[0])
np.random.shuffle(randomize)

X_train = trainImages[randomize]
y_train = trainLabels[randomize].flatten()

randomizeVal = np.arange(valImages.shape[0])
X_val = valImages[randomizeVal]
y_val = valLabels[randomizeVal].flatten()

randomizeTest = np.arange(testImages.shape[0])
X_test = testImages[randomizeTest]
y_test = testLabels[randomizeTest].flatten()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

print('first 10 examples in train: ',y_train[:10])

num_val = 500
mask = list(range(num_val))
X_val = X_val[mask]
y_val = y_val[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

print('first 10 examples in val: ',y_val[:10])

# Reshape the image data into rows for effecient distance calculation
#(vi tager billedet med dimensioner 28x28x3 og strækker det ud til en vektor med længden 28*28*3 = 2352)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(f'New train shape: {X_train.shape}')
print(f'New val shape: {X_val.shape}')

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((28,28,3)).astype('uint8')) # visualize the mean image
plt.show()

# second: subtract the mean image from train and test data
X_train = X_train.astype(np.float32)-mean_image
X_val = X_val.astype(np.float32)-mean_image
X_test = X_test.astype(np.float32)-mean_image

print(X_train.min(),X_train.max())

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print(X_train.shape, X_val.shape)
'''
image_size = 28*28*3+1
num_classes = 8

W = np.random.randn(image_size, num_classes) * 0.0001

print(dl.svm_loss(W, X_train, y_train, 1)[1][0])

classifier = LinearClassifier(input_dim=28*28*3+1,num_classes=8, loss_type=Loss_type.SOFT_MAX)

history = classifier.train(X=X_train, y=y_train, num_iters=5000)

plt.plot(history)
plt.show()

    #make predictions on the validation data
predictions = classifier.predict(X=X_val)

    #calculate the accuracy

num_correct = sum(predictions == y_val)

print((num_correct/num_val)*100)
'''

best_svm = None
best_accuracy = 0

learning_rates = [1e-2,1e-4,1e-6]
regularization_strenghts = [1e-2, 1e-1, 1, 1e2]

num_iters = 1000
batch_size = 200

for lr in learning_rates:
    for reg in regularization_strenghts:
        print(f"learning rate {lr}")
        print(f"reg strength {reg}")
        
        svm = LinearClassifier(input_dim=28*28*3+1,num_classes=8,loss_type=Loss_type.SVM)

        history_loss = svm.train(X=X_train, y=y_train, learning_rate=lr, reg=reg, num_iters=2500)
        plt.plot(history_loss)

        pred = svm.predict(X=X_val)
        accuracy = (sum(pred == y_val) / num_val) * 100

        if accuracy > best_accuracy:
            best_svm = svm
            best_accuracy = accuracy

print(f"best_accuracy during validation {best_accuracy}")
            
plt.show()

test = best_svm.predict(X=X_test)
test_accuracy = (sum(test == y_test) / num_test) * 100

print(f"final final final test accuracy {test_accuracy}")