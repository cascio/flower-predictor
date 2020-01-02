import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label


export_path_keras = './1577938401.h5'

reloaded = tf.keras.models.load_model(
    export_path_keras,
    custom_objects={'KerasLayer': hub.KerasLayer})

reloaded.summary()

splits = tfds.Split.TRAIN.subsplit([70, 30])

(training_set, validation_set), dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True, split=splits) # noqa

num_classes = dataset_info.features['label'].num_classes

num_training_examples = 0
num_validation_examples = 0

for example in training_set:
    num_training_examples += 1

for example in validation_set:
    num_validation_examples += 1

IMAGE_RES = 224
BATCH_SIZE = 30

train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1) # noqa

validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1) # noqa

class_names = np.array(dataset_info.features['label'].names)

print(class_names)


for i in range(4):
    print(i)
    image_batch, label_batch = next(iter(train_batches))

    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()

    predicted_batch = reloaded.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()

    predicted_ids = np.argmax(predicted_batch, axis=-1)
    predicted_class_names = class_names[predicted_ids]

    print(predicted_class_names)

    print("Labels:           ", label_batch)
    print("Predicted labels: ", predicted_ids)

    correct = (predicted_ids == label_batch)
    accuracy = correct.sum() / correct.size
    print(accuracy)

    plt.figure(figsize=(10, 9))
    for n in range(30):
        plt.subplot(6, 5, n+1)
        plt.subplots_adjust(hspace=0.3)
        plt.imshow(image_batch[n])
        color = "blue" if predicted_ids[n] == label_batch[n] else "red"
        plt.title(predicted_class_names[n].title(), color=color)
        plt.axis('off')
    _ = plt.suptitle(f'Model predictions (blue: correct, red: incorrect) | Accuracy: {accuracy}') # noqa

    plt.savefig(f'{i}.png')
