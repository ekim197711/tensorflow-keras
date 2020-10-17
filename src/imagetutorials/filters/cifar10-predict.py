import tensorflow as tf
import tensorflow.keras.preprocessing.image as image

class_names = ['airplane','automobile','bird','cat','deer','dog','frog',
               'horse','ship','truck'
]

def predictAnImage(imagelocation):
    model = tf.keras.models.load_model('./mycifar10.model')
    model.summary()
    loadedimg =image.load_img(imagelocation,  target_size=(32, 32))
    origarray = image.img_to_array(loadedimg, dtype=float)
    # print(f'{origarray.shape}')
    input_arr = origarray.astype(float) / 255.0
    input_arr = input_arr.reshape(1, 32, 32, 3)

    print(f'{input_arr.shape}')
    result = model.predict(input_arr)
    result2 = model.predict_classes(input_arr)
    print(f'result predict: {result[0]}')
    print(f'result2 predict: {class_names[result2[0]]}')

# predictAnImage('./frog1.png')
# predictAnImage('./truck1.png')
predictAnImage('./horse1.png')