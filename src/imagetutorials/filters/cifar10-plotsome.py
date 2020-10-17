# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
print(f'{train_images.shape}')
pageno = 5
rangestart = 9 * pageno
rangeend = rangestart+9
for i in range(rangestart, rangeend):
    # define subplot
    index = 1 + i-rangestart;
    plt.subplot(3,3, index)
    # plot raw pixel data
    plt.imshow(train_images[i])
# show the figure
plt.show()
