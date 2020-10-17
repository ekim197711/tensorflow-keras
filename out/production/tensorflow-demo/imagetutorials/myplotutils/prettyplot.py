import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt

def plot_image_custom(predictions_array, img, class_names):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                         100*np.max(predictions_array)),
               fontsize=18,color='blue',)



def plot_value_array_custom(predictions_array, class_names):
    plt.grid(False)
    plt.xticks(ticks=range(10), labels=class_names
               , fontsize=16
               ,rotation='vertical')
    plt.yticks(ticks=arange(0.0,1.0,0.1))

    # print(f'pred array {predictions_array}')
    thisplot = plt.bar(range(10), predictions_array,  color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('green')
