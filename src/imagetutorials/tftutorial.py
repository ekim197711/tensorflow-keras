import tensorflow as tf
print(f'Tf version {tf.version.VERSION}')


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f'X train data shape {x_train.shape}')
print(f'Y train data shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'Y test data shape {y_test.shape}')
onetensor = x_train[4:5]
print(f'onetensor {onetensor}')


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)

model.summary()

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

model.save('./models/mymnist1.model')





