import random
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# NEW: keep track of which indices were picked per class, so we can reuse them later
class_selected_indices = {}

for class_,name in trainInfo.items():
    print(f'Class: {class_}, class name: {name}. Number of training samples: {len(trainLabels[trainLabels==int(class_)])}')
    # Get indices of all images belonging to class i
    class_indices = [idx for idx, label in enumerate(trainLabels) if int(class_) == label]
    # Randomly select 5 indices
    selected_indices = random.sample(class_indices, 5)
    class_selected_indices[int(class_)] = selected_indices  # NEW: store for reuse
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
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(f'New train shape: {X_train.shape}')
print(f'New val shape: {X_val.shape}')

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
print(mean_image[:10]) # print a few of the elements

mean_image_reshaped = mean_image.reshape((28, 28, 3))  # NEW: reusable reshaped version

plt.figure(figsize=(4,4))
plt.imshow(mean_image_reshaped.astype('uint8')) # visualize the mean image
plt.title('Mean image')
plt.show()

# NEW: plot the same sample images as before, but mean-subtracted
fig, axes = plt.subplots(5, len(trainInfo), figsize=(15, 5))

for class_, name in trainInfo.items():
    selected_indices = class_selected_indices[int(class_)]  # reuse same indices as first plot
    for j, idx in enumerate(selected_indices):
        image = trainImages[idx].astype(np.float32) - mean_image_reshaped
        # normalize for display: shift/scale so values fall roughly in [0, 1]
        image_display = (image - image.min()) / (image.max() - image.min())
        axes[j, int(class_)].imshow(image_display)
        axes[j, int(class_)].axis('off')
        if j == 0:
            axes[j, int(class_)].set_title(f'{name[:5]}: {class_}')

plt.suptitle('Mean-subtracted samples')
plt.tight_layout()
plt.show()

# second: subtract the mean image from train and test data
X_train = X_train.astype(np.float32)-mean_image
X_val = X_val.astype(np.float32)-mean_image
X_test = X_test.astype(np.float32)-mean_image

print(X_train.min(),X_train.max())

print(X_train.shape, X_val.shape)

# Small toy array so every value is visible and traceable
np.random.seed(1)
H, W, C = 3, 3, 3  # tiny 3x3 image, 3 channels
toy_array = np.random.randint(0, 10, size=(H, W, C))

flattened = toy_array.flatten()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# --- LEFT: show the 3D array as 3 separate H x W grids (one per channel) ---
ax = axes[0]
ax.set_xlim(0, W * (C) + (C - 1))
ax.set_ylim(0, H)
ax.invert_yaxis()
ax.axis('off')
ax.set_title(f'Original array, shape ({H}, {W}, {C})', fontsize=13)

channel_colors = ['#ffb3b3', '#b3ffb3', '#b3b3ff']  # light red, green, blue
channel_labels = ['R', 'G', 'B']

cell_size = 1
gap = 1  # gap between channel blocks

for c in range(C):
    x_offset = c * (W + gap)
    for i in range(H):
        for j in range(W):
            x = x_offset + j
            y = i
            rect = patches.Rectangle((x, y), cell_size, cell_size,
                                      facecolor=channel_colors[c], edgecolor='black')
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.5, str(toy_array[i, j, c]),
                    ha='center', va='center', fontsize=11)
    ax.text(x_offset + W/2, -0.4, f'Channel {channel_labels[c]}', ha='center', fontsize=11)

# --- RIGHT: show the flattened 1D array, colored to match source channel ---
ax2 = axes[1]
ax2.set_xlim(0, len(flattened))
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title(f'Flattened array, shape ({H*W*C},)', fontsize=13)

idx = 0
for i in range(H):
    for j in range(W):
        for c in range(C):
            rect = patches.Rectangle((idx, 0), 1, 1,
                                      facecolor=channel_colors[c], edgecolor='black')
            ax2.add_patch(rect)
            ax2.text(idx + 0.5, 0.5, str(toy_array[i, j, c]),
                    ha='center', va='center', fontsize=9, rotation=90)
            idx += 1

plt.tight_layout()
plt.savefig('flatten_mechanics.png', dpi=150, bbox_inches='tight')
plt.show()

print("Original array:\n", toy_array)
print("\nFlattened array:\n", flattened)