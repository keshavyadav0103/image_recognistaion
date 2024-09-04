import tensorflow_datasets as tfds
import tensorflow as tf

# Load the COCO dataset
(ds_train, ds_test), ds_info = tfds.load(
    'coco/2017',
    split=['train', 'validation'],
    shuffle_files=True,
    with_info=True,
    as_supervised=True,  # This returns (image, label) pairs
)

# Define a function to preprocess the images and labels
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))  # Resize image to match model input
    label = tf.one_hot(label, ds_info.features['objects']['label'].num_classes)
    return image, label

# Apply the preprocessing to the datasets
ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache().shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
