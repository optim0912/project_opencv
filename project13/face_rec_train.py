import io
import os
import argparse
import datetime
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import plot_model

# Default paths.
SCRIPT_PATH = Path(__file__).resolve().parent
DEFAULT_TFRECORDS_DIR = str(SCRIPT_PATH / 'tfrecords-output')
DEFAULT_NUM_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_BATCH_SIZE = 30
DEFAULT_SAVE_PERIOD = 10

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 200
IMAGE_CHANNEL = 1
args = None

num_classes = 5

#tf.get_logger().setLevel('ERROR')
#tf.debugging.set_log_device_placement(False)

def _parse_function(example):
    features = tf.io.parse_single_example(
        example,
        features={
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string,
                                                default_value='')
        })
    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.io.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

    return image, label

def augment(image_label, seed):
    image, label = image_label
    image = tf.image.resize_with_crop_or_pad(image, IMAGE_HEIGHT + 6, IMAGE_WIDTH + 6)
    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=3)
    # Random crop back to the original size
    image = tf.image.stateless_random_crop(
        image, size=[IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], seed=new_seed[0])
    # Random flip
    image = tf.image.stateless_random_flip_left_right(image, seed=new_seed[1])
    # Random brightness
    image = tf.image.stateless_random_brightness(
        image, max_delta=0.5, seed=new_seed[2])
    image = tf.clip_by_value(image, 0, 1)
    return image, label

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt, manager, period):
        super(MyCallback, self).__init__()
        self.period = period
        self.ckpt = ckpt
        self.manager = manager
    
    def on_epoch_end(self, epoch, logs=None):
        self.ckpt.epoch.assign_add(1)
        if int(self.ckpt.epoch) % self.period == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.epoch), save_path))

def MyModel():
    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=L2(0.0001))(inputs)
    x = MaxPool2D(2, strides=2)(x)
    x = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=L2(0.0001))(x)
    x = MaxPool2D(2, strides=2)(x)
    x = Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=L2(0.0001))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=L2(0.0001))(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)

    model = Model(inputs, x, name="face_rec_cnn")
    return model

def main():

    print('Processing data...')

    tf_records_dir_path = Path(args.tfrecords_dir)
    train_data_files = list(map(str, tf_records_dir_path.glob('train-*')))
    val_data_files = list(map(str, tf_records_dir_path.glob('test-*')))

    # Create training dataset input pipeline.
    counter = tf.data.experimental.Counter()
    train_dataset = tf.data.TFRecordDataset(train_data_files) \
        .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = tf.data.Dataset.zip((train_dataset, (counter, counter))) \
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .shuffle(1000) \
        .batch(args.batch_size) \
        .prefetch(tf.data.AUTOTUNE)

    # Create testing dataset input pipeline.
    val_dataset = tf.data.TFRecordDataset(val_data_files) \
        .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(args.batch_size) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)

    # Create the model!
    model = MyModel()
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    model.summary()
    probability_model.summary()
    plot_model(model, to_file="face_rec_cnn1.png", show_shapes=True)
    plot_model(probability_model, to_file="face_rec_cnn2.png", show_shapes=True)

    # Define our loss.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss = SparseCategoricalCrossentropy(from_logits=True),
        metrics = [SparseCategoricalAccuracy()]
    )

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), model=model)
    manager = tf.train.CheckpointManager(ckpt, 'training_face_rec_cnn', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    log_dir = "logs/face_rec_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    my_callback = MyCallback(ckpt, manager, args.save_period)
    early_callback = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_sparse_categorical_accuracy",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=5,
        verbose=1
    )
    ] 

    print('Start learning!')
    model.fit(
        train_dataset,
        epochs = args.num_train_epochs,
        callbacks = [my_callback, tensorboard_callback, early_callback],
        initial_epoch = int(ckpt.epoch),
        validation_data = val_dataset
    )
    print('Learning finished!')

    '''
    model.evaluate(
        test_dataset
    )
    '''

    tf.saved_model.save(probability_model, 'face_rec_cnn')
    print('face_rec_cnn.pb file is created successfully!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                        default=DEFAULT_TFRECORDS_DIR,
                        help='Directory of TFRecords files.')
    parser.add_argument('--num-train-epochs', type=int,
                        dest='num_train_epochs',
                        default=DEFAULT_NUM_EPOCHS,
                        help='Number of times to iterate over all of the '
                             'training data.')
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        default=DEFAULT_LEARNING_RATE,
                        help='How large a learning rate to use when training.')
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',
                        default=DEFAULT_BATCH_SIZE,
                        help='How many images to train on at a time.')
    parser.add_argument('--save-period', type=int,
                        dest='save_period',
                        default=DEFAULT_SAVE_PERIOD,
                        help='How many epochs to save ckpt files.')
    args = parser.parse_args()
    main()