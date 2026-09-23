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
from simple_neural_network import FullyConnectedNN

def random_search(X_train, y_train, X_val, y_val, num_trials=20):
    best_acc = 0
    best_params = None
    best_model = None

    for trial in range(num_trials):
        # Sample hyperparameters from a reasonable range (log scale for lr and reg!)
        lr = 10 ** np.random.uniform(-5, -1)        # e.g. 1e-5 to 1e-1
        reg = 10 ** np.random.uniform(-5, 1)         # e.g. 1e-5 to 1e1
        hidden_size = np.random.choice([64, 128, 256, 512])
        batch_size = np.random.choice([32, 64, 128])
        optimizer = np.random.choice(['sgd', 'momentum', 'adam'])

        layers = [X_train.shape[1], hidden_size, 8]  # adjust output size to your num_classes

        model = FullyConnectedNN(layers=layers, reg_strength=reg, loss='softmax')
        model.fit(X_train, y_train, epochs=20, batch_size=batch_size,
                  learning_rate=lr, optimizer=optimizer)

        preds = model.predict(X_val)
        acc = np.mean(preds == y_val)

        print(f"trial {trial}: lr={lr:.5f}, reg={reg:.5f}, hidden={hidden_size}, "
              f"batch={batch_size}, opt={optimizer} -> val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = dict(lr=lr, reg=reg, hidden_size=hidden_size,
                                batch_size=batch_size, optimizer=optimizer)
            best_model = model

    return best_model, best_params, best_acc

trainDataset = BloodMNIST(split="train", download=True,size=28)
valDataset = BloodMNIST(split="val", download=True,size=28)
testDataset = BloodMNIST(split="test", download=True,size=28)

trainImages = trainDataset.imgs
trainLabels = trainDataset.labels
trainInfo = trainDataset.info.get('label')

print('Training data:')
print(f'Images: {trainImages.shape}, Labels: {trainLabels.shape}')

valImages = valDataset.imgs
valLabels = valDataset.labels
print('Validation data:')
print(f'Images: {valImages.shape}, Labels: {valLabels.shape}')

testImages = testDataset.imgs
testLabels = testDataset.labels
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
num_training = 5950
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

print('first 10 examples in train: ',y_train[:10])

num_val = 850
mask = list(range(num_val))
X_val = X_val[mask]
y_val = y_val[mask]

num_test = 1700
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

print(X_train.shape, X_val.shape)

X_train /= float(255)
X_val /= float(255)

#nn = FullyConnectedNN(layers=[X_train.shape[1], 500, 8], loss='softmax')
#nn.fit(X_train, y_train,learning_rate=0.01, epochs=200,optimizer='sgd')

#preds = nn.predict(X_train)
#print(f'Training accuracy={np.mean(preds==y_train)}')


#val_preds = nn.predict(X_val)
#print(f'Validation accuracy={np.mean(val_preds==y_val)}')

# Finding the best params
best_model, best_params, best_acc = random_search(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, num_trials=20)

print(f"Best validation accuracy during optimization {best_acc}")
test_preds_after_optimization = best_model.predict(X_test)
print(f"Test Accuracy after random optimization{np.mean(test_preds_after_optimization==y_test)}")

for k,v in best_params.items():
    print(k, v)
