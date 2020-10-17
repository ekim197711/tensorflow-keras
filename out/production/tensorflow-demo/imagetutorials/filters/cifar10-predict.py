import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as image

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
print(f'{train_images.shape}')
print(f'{train_images[:1]}')
class_names = ['airplane','automobile','bird','cat','deer','dog','frog',
               'horse','ship','truck'
]

model = tf.keras.models.load_model('./mycifar10.model')
model.summary()
frogimg =image.load_img("./frog1.png",  target_size=(32, 32))
origarray = image.img_to_array(frogimg, dtype=float)
print(f'{origarray.shape}')
input_arr = origarray / 255.0
input_arr = input_arr.reshape(1, 32, 32, 3)

print(f'{input_arr.shape}')
result = model.predict(input_arr)
result2 = model.predict_classes(input_arr)
print(f'result predict: {result}')
print(f'result2 predict: {class_names[result2[0]]}')

# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# pretty.plot_image_custom(result, origarray, class_names)
# plt.subplot(1,2,2)
# pretty.plot_value_array_custom(result[0], class_names)
# plt.show()


