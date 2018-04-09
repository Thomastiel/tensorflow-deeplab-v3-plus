"""Train a DeepLab v3 plus model using tf.estimator API."""

import argparse
import os
import sys
import glob

import tensorflow as tf
tfexample_decoder = tf.contrib.slim.tfexample_decoder

import deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug

import shutil

default_data_path = "/mnt/data/users/thomas/garden/Tensorflow"


parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default='test',
                    help='Run name.')

parser.add_argument('--gpu', type=str, default='0',
                    help='GPU.')

parser.add_argument('--model_dir', type=str, default=default_data_path + '/train',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=20,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=10,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

# parser.add_argument('--data_dir', type=str, default='/home/thomas/data-ssd/garden-tf/tfrecord/rgb/',
#                     help='Path to the directory containing the PASCAL VOC data tf record.')
parser.add_argument('--data_dir', type=str, default=default_data_path + '/tfrecord-fix/rgb/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--pre_trained_model', type=str,
                    default='/mnt/data/users/thomas/cityscapes/pretrained/resnet_v2_101.ckpt',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 16
_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 0

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

_NUM_IMAGES = {
    'train': 33677,
    'validation': 4812,
}

def get_filenames(is_training, data_dir):
    """Return a list of filenames.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: path to the the directory containing the input data.

    Returns:
      A list of file names.
    """
    if is_training:
        pattern = 'train-*'
    else:
        pattern = 'val-*'
    files = glob.glob(os.path.join(data_dir, pattern))
    return files


def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/segmentation/class/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation/class/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
    }

    items_to_handlers = {
        'image': tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            channels=3),
        'image_name': tfexample_decoder.Tensor('image/filename'),
        'height': tfexample_decoder.Tensor('image/height'),
        'width': tfexample_decoder.Tensor('image/width'),
        'label': tfexample_decoder.Image(
            image_key='image/segmentation/class/encoded',
            format_key='image/segmentation/class/format',
            channels=1),
    }


    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    image, label = decoder.decode(raw_record, ['image', 'label'])

    # parsed = tf.parse_single_example(raw_record, keys_to_features)
    #
    # # height = tf.cast(parsed['image/height'], tf.int32)
    # # width = tf.cast(parsed['image/width'], tf.int32)
    #
    # image = tf.image.decode_image(
    #     tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
    # image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    # image.set_shape([None, None, 3])
    #
    # label = tf.image.decode_image(
    #     tf.reshape(parsed['label/encoded'], shape=[]), 1)
    # label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    # label.set_shape([None, None, 1])

    return image, label


def preprocess_image(image, label, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    # if is_training:
        # Randomly scale the image and label.
        # image, label = preprocessing.random_rescale_image_and_label(
        #     image, label, _MIN_SCALE, _MAX_SCALE)
        #
        # # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        # image, label = preprocessing.random_crop_or_pad_image_and_label(
        #     image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)
        #
        # # Randomly flip the image and label horizontally.
        # image, label = preprocessing.random_flip_left_right_image_and_label(
        #     image, label)
        # image.set_shape([_HEIGHT, _WIDTH, 3])
        # label.set_shape([_HEIGHT, _WIDTH, 1])

    # image = preprocessing.mean_image_subtraction(image)
    # image = tf.transpose(image, [2, 0, 1])
    # label = tf.transpose(label, [2, 0, 1])

    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int32)

    return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      A tuple of images and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
        lambda image, label: preprocess_image(image, label, is_training))
    dataset = dataset.prefetch(batch_size)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels

def validate_dataset(filenames, reader_opts=None):
    """
    Attempt to iterate over every record in the supplied iterable of TFRecord filenames
    :param filenames: iterable of filenames to read
    :param reader_opts: (optional) tf.python_io.TFRecordOptions to use when constructing the record iterator
    """
    i = 0
    for fname in filenames:
        print('validating ', fname)

        record_iterator = tf.python_io.tf_record_iterator(path=fname, options=reader_opts)
        try:
            for _ in record_iterator:
                i += 1
        except Exception as e:
            print('error in {} at record {}'.format(fname, i))
            print(e)

def main(unused_argv):
    # validate_dataset(get_filenames(True, FLAGS.data_dir))
    # validate_dataset(get_filenames(False, FLAGS.data_dir))
    # return

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    model_dir = os.path.join(FLAGS.model_dir, FLAGS.name)

    if FLAGS.clean_model_dir:
        shutil.rmtree(model_dir, ignore_errors=True)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    model = tf.estimator.Estimator(
        model_fn=deeplab_model.deeplabv3_plus_model_fn,
        model_dir=model_dir,
        config=run_config,
        params={
            'output_stride': FLAGS.output_stride,
            'batch_size': FLAGS.batch_size,
            'base_architecture': FLAGS.base_architecture,
            'pre_trained_model': FLAGS.pre_trained_model,
            'batch_norm_decay': _BATCH_NORM_DECAY,
            'num_classes': _NUM_CLASSES,
            'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
            'weight_decay': FLAGS.weight_decay,
            'learning_rate_policy': FLAGS.learning_rate_policy,
            'num_train': _NUM_IMAGES['train'],
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'max_iter': FLAGS.max_iter,
            'end_learning_rate': FLAGS.end_learning_rate,
            'power': _POWER,
            'momentum': _MOMENTUM,
            'freeze_batch_norm': FLAGS.freeze_batch_norm,
            'initial_global_step': FLAGS.initial_global_step
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_px_accuracy': 'train_px_accuracy',
            'train_mean_iou': 'train_mean_iou',
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)
        train_hooks = [logging_hook]
        eval_hooks = None

        if FLAGS.debug:
            debug_hook = tf_debug.LocalCLIDebugHook()
            train_hooks.append(debug_hook)
            eval_hooks = [debug_hook]

        tf.logging.info("Start training.")
        model.train(
            input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=train_hooks,
            # steps=1  # For debug
        )

        tf.logging.info("Start evaluation.")
        # Evaluate the model and print results
        eval_results = model.evaluate(
            # Batch size must be 1 for testing because the images' size differs
            input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
            hooks=eval_hooks,
            steps=100  # For debug
        )
        print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
