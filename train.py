import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import time

from tensorflow.keras import layers


# Download Flowers Dataset using TensorFlow Datasets & split into training & validation sets using a 75%-25% split # noqa
def split_data(dataset, splits):
    splits = tfds.Split.TRAIN.subsplit(splits)
    (training_set, validation_set), dataset_info = tfds.load(dataset, with_info=True, as_supervised=True, split=splits) # noqa
    return (training_set, validation_set), dataset_info


# Reformats images to (224,224) and normalizes pixel values between 0-1
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


splits = [75, 25]
dataset = 'tf_flowers'
IMAGE_RES = 224
BATCH_SIZE = 32

(training_set, validation_set), dataset_info = split_data(dataset, splits)
num_classes = dataset_info.features['label'].num_classes
num_training_examples = 0
num_validation_examples = 0

for example in training_set:
    num_training_examples += 1

for example in validation_set:
    num_validation_examples += 1

print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples)) # noqa

train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1) # noqa
validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1) # noqa

# Create a keras layer using the pre-trained MobileNetv2 model, without the final classification layer # noqa
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

# Freeze the pre-trained model
feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train the model
EPOCHS = 6
history = model.fit(
    train_batches,
    epochs=EPOCHS,
    validation_data=validation_batches
)

# Save as Keras .h5 model
t = time.time()
export_path_keras = "./{}.h5".format(int(t))
model.save(export_path_keras)
print(export_path_keras)
