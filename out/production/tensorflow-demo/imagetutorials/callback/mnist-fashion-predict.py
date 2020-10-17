import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import imagetutorials.myplotutils.prettyplot as pretty
import tensorflow.keras.preprocessing.image as image
import imagetutorials.callback.mycallback as mycallback

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.models.load_model('../../models/myfashion.model')
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
mikesimg =image.load_img("mikesimg.png", color_mode="grayscale", target_size=(28, 28))
origarray = image.img_to_array(mikesimg, dtype=float)
input_arr = origarray / 255.0
input_arr = tf.transpose(input_arr, [2,0,1])

result = probability_model.predict(input_arr, callbacks=[mycallback.MyCallback()])
print(f'result predict: {result}')

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
pretty.plot_image_custom(result, origarray, class_names)
plt.subplot(1,2,2)
pretty.plot_value_array_custom(result[0], class_names)
plt.show()


