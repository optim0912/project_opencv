#!/usr/bin/env python

import argparse
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import datetime
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

training_epochs = 15
BATCH_SIZE = 100
learning_rate = 0.001
save_period = 1

# Default paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  './labels/2350-common-hangul.txt')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'saved-model')
DEFAULT_NUM_EPOCHS = training_epochs

# This will be determined by the number of entries in the given label file.
num_classes = 2350

#tf.get_logger().setLevel('ERROR')
#tf.debugging.set_log_device_placement(False)

def _parse_function(example):
    features = tf.io.parse_single_example(
        example,
        features={
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })
    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.io.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes))
    return image, label

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt, manager, period):
        super(MyCallback, self).__init__()
        self.period = period
        self.ckpt = ckpt
        self.manager = manager
    
    def on_epoch_end(self, epoch, logs=None):
        self.ckpt.epoch.assign_add(1)
        if int(self.ckpt.epoch) % save_period == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.epoch), save_path))

def MyModel():
    inputs = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    x = Conv2D(32, 5, padding='same', activation='relu')(inputs)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    x = Conv2D(64, 5, padding='same', activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(2, strides=2, padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(num_classes)(x)

    model = Model(inputs, x, name="hangul_cnn")
    return model

def main(label_file, tfrecords_dir, model_output_dir, num_train_epochs):
    """Perform graph definition and model training.

    Here we will first create our input pipeline for reading in TFRecords
    files and producing random batches of images and labels.
    Next, a convolutional neural network is defined, and training is performed.
    After training, the model is exported to be used in applications.
    """
    global num_classes
    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()
    num_classes = len(labels)

    print('Processing data...')

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    train_data_files = tf.io.gfile.glob(tf_record_pattern)

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.io.gfile.glob(tf_record_pattern)

    # Create training dataset input pipeline.
    train_dataset = tf.data.TFRecordDataset(train_data_files) \
        .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .cache() \
        .shuffle(1000) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Create testing dataset input pipeline.
    test_dataset = tf.data.TFRecordDataset(test_data_files) \
        .map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(BATCH_SIZE) \
        .cache() \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Create the model!
    model = MyModel()
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    model.summary()
    probability_model.summary()
    plot_model(model, to_file="hangul_cnn1.png", show_shapes=True)
    plot_model(probability_model, to_file="hangul_cnn2.png", show_shapes=True)

    # Define our loss.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss = CategoricalCrossentropy(from_logits=True),
        metrics = [CategoricalAccuracy()]
    )

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), model=model)
    manager = tf.train.CheckpointManager(ckpt, 'training_hangul_cnn', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    log_dir = "logs/hangul_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    my_callback = MyCallback(ckpt, manager, save_period)

    print('Start learning!')
    model.fit(
        train_dataset,
        epochs = training_epochs,
        callbacks = [my_callback, tensorboard_callback],
        initial_epoch = int(ckpt.epoch)
    )
    print('Learning finished!')

    model.evaluate(
        test_dataset
    )

    tf.saved_model.save(probability_model, 'hangul_cnn')
    print('hangul_cnn.pb file is created successfully!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                        default=DEFAULT_TFRECORDS_DIR,
                        help='Directory of TFRecords files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store saved model files.')
    parser.add_argument('--num-train-epochs', type=int,
                        dest='num_train_epochs',
                        default=DEFAULT_NUM_EPOCHS,
                        help='Number of times to iterate over all of the '
                             'training data.')
    args = parser.parse_args()
    main(args.label_file, args.tfrecords_dir,
         args.output_dir, args.num_train_epochs)