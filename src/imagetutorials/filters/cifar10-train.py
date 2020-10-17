# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 0: airplane# 1: automobile#
# 2: bird# 3: cat#
# 4: deer# 5: dog#
# 6: frog # 7: horse#
# 8: ship# 9: truck

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
print(f'{train_images.shape}')

train_images_input = train_images / 255.0
test_images_input = test_images /  255.0
# print(f'{train_images[:1]}')
# print(f'{train_images_input[:1]}')

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    # keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation='softmax')
])
# opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images_input, train_labels, epochs=20, batch_size=50)
model.summary()
model.save("./mycifar10.model")
test_loss, test_acc = model.evaluate(test_images_input,  test_labels, verbose=2)
