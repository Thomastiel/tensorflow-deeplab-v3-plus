# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts Cityscapes data to TFRecord file format with Example protos.

The Cityscapes dataset is expected to have the following directory structure:

  + cityscapes
     - build_cityscapes_data.py (current working directiory).
     - build_data.py
     + cityscapesscripts
       + annotation
       + evaluation
       + helpers
       + preparation
       + viewer
     + gtFine
       + train
       + val
       + test
     + leftImg8bit
       + train
       + val
       + test
     + tfrecord

This script converts data into sharded data files and save at tfrecord folder.

Note that before running this script, the users should (1) register the
Cityscapes dataset website at https://www.cityscapes-dataset.com to
download the dataset, and (2) run the script provided by Cityscapes
`preparation/createTrainIdLabelImgs.py` to generate the training groundtruth.

Also note that the tensorflow model will be trained with `TrainId' instead
of `EvalId' used on the evaluation server. Thus, the users need to convert
the predicted labels to `EvalId` for evaluation on the server. See the
vis.py for more details.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import glob
import math
import os.path
import re
import sys
import build_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('garden_root',
                           '/mnt/data/users/thomas/garden/',
                           'Cityscapes dataset root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    'Tensorflow/tfrecord-full/',
    'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 20

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'Tensorflow/mat_export',
    'albedo': 'Tensorflow/mat_export',
    'shading': 'Tensorflow/mat_export',
    'segmentation': 'anno-label-clean',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_rgb',  # or _albedo
    'albedo': '_albedo',  # or _albedo
    'shading': '_shading',  # or _albedo
    'segmentation': '',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'albedo': 'png',
    'shading': 'png',
    'segmentation': 'png',
}

_SPLIT_LIST = {
    'train': 'Split/train.txt',
    'val': 'Split/val.txt',
    'test': 'Split/test.txt',
}

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('((([^\/]*)\/){2}[^\/]*)' + _POSTFIX_MAP['image'])


def _get_files(data, dataset_split):
    """Gets files for the specified data type and dataset split.

    Args:
      data: String, desired data ('image' or 'label').
      dataset_split: String, dataset split ('train', 'val', 'test')

    Returns:
      A list of sorted file names or None when getting label for
        test set.
    """
    # if data == 'label' and dataset_split == 'test':
    #   return None

    with open(os.path.join(FLAGS.garden_root, _SPLIT_LIST[dataset_split])) as f:
        filenames = [l.split()[0] for l in f.readlines()]
        filenames = [file.replace('.png', '%s.png' % _POSTFIX_MAP[data]) for file in filenames]
        filenames = [os.path.join(FLAGS.garden_root, _FOLDERS_MAP[data], *file.split('/')[-3:]) for file in filenames]
    # print(data, filenames[0])

    return sorted(filenames)


def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.

    Args:
      dataset_split: The dataset split (e.g., train, val).

    Raises:
      RuntimeError: If loaded image and label have different shape, or if the
        image file with specified postfix could not be found.
    """
    image_files = _get_files('image', dataset_split)
    albedo_files = _get_files('albedo', dataset_split)
    shading_files = _get_files('shading', dataset_split)
    segmentation_files = _get_files('segmentation', dataset_split)

    num_images = len(image_files)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('png', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        shard_filename = '%s-%05d-of-%05d.tfrecord' % (
            dataset_split, shard_id, _NUM_SHARDS)
        output_filename = os.path.join(FLAGS.garden_root, FLAGS.output_dir, shard_filename)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image.
                image_data = tf.gfile.FastGFile(image_files[i], 'rb').read()
                albedo_data = tf.gfile.FastGFile(albedo_files[i], 'rb').read()
                shading_data = tf.gfile.FastGFile(shading_files[i], 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                # Read the semantic segmentation annotation.
                seg_data = tf.gfile.FastGFile(segmentation_files[i], 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)

                if height != seg_height or width != seg_width:
                    print(image_files[i])
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                re_match = _IMAGE_FILENAME_RE.search(image_files[i])
                if re_match is None:
                    raise RuntimeError('Invalid image filename: ' + image_files[i])
                filename = os.path.basename(re_match.group(1)).encode()

                example = build_data.garden_image_to_tfexample(
                    image_data, albedo_data, shading_data, seg_data, filename, height, width)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Only support converting 'train' and 'val' sets for now.
    for dataset_split in ['train', 'val']:
        print(dataset_split)
        _convert_dataset(dataset_split)


if __name__ == '__main__':
    tf.app.run()
