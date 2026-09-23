from medmnist import BloodMNIST
import numpy as np


def printLabels(labels, split):
    classDict = {}
    print(split)
        
    for label in labels.flatten():
        label = int(label)
        classDict[label] = classDict.get(label, 0) + 1
        
    for k,v in sorted(classDict.items()):
        print(k,v)
    

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

printLabels(trainLabels, "training split")
printLabels(valLabels, "val split")
printLabels(testLabels, "test split")