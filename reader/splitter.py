import os
import sys
import numbers

from six.moves import urllib

import numpy as np
import tensorflow as tf


class ClassSplitter(object):
    """
    Currently only intended for images. (HxWxD)

    Dev notes: (cmb)
     It might be better to make this an abstract class.
     Then we can force people to define the subclass for 
     each dataset instead of populating the single class
     like I'm doing now.
    """

    def __init__(self):

        self.name = ""

        self.height = 32
        self.width  = 32
        self.depth  = 3

        # It might be better to specify the data type
        # then we could calculate it internally rather 
        # than specify it manually
        self.bytes_per_ex = 3073

        self.raw_dir = ""
        self.tfr_dir = ""

        self.total_training = 0
        self.training_files = ""
        self.training_tfr   = ""

        self.total_validation   = 0
        self.validation_files   = ""
        self.validation_tfr     = ""

        self.total_testing = 0
        self.testing_files = ""
        self.testing_tfr   = ""

        # This is intended as a mechanism for creating 
        # splits when we only get one input set
        # Not implemented
        self.validation_percent = 0
        self.testing_percent = 0

        self.join_out_classes = True
        self._raw_class_num = 2
        self._raw_in_classes = [0, 1]
        self._set_map()

        self.raw_class_names = {}
        self.raw_subclass_names = None

        self.auto_download = True
        self.websites = ""

        self.raw_reader = None
        self.extractor  = None

        # parser is a function that returns data, label
        self.parser = lambda ex: parser(ex, 
            self.height, self.width, self.depth)
        # augmenter is a function that augments the training data
        self.augmenter = lambda ex: ex

        self.num_parallel_calls = 64
        self.buffer_size = 4096


    @property
    def raw_in_classes(self):
        # This is a property so the map gets updated automatically
        # and some checks/sorting are performed
        return self._raw_in_classes
    
    @raw_in_classes.setter
    def raw_in_classes(self, value):
        if isinstance(value, numbers.Number) or \
            isinstance(value, list) or \
            isinstance(value, np.ndarray):
            value = list(value)
        else:
            raise TypeError("The in-class cases must be numeric.")
        value.sort()

        assert all(isinstance(x, numbers.Number) for x in value)
        assert all(x is not None for x in value)

        self._raw_in_classes = value
        self._set_map()
        # self._update_dict()


    @property
    def raw_class_num(self):
        # This is a property so the map gets updated automatically
        return self._raw_class_num
    
    @raw_class_num.setter
    def raw_class_num(self, value):
        if not self._raw_class_num == value:
            self._raw_class_num = value
            self._set_map()
            # self._update_dict()


    @property
    def raw_all_classes(self):
        # Convenience
        return np.array([ii for ii in range(self.raw_class_num)])
    

    @property
    def raw_out_classes(self):
        if self.join_out_classes:
            N = self.new_class_num
            out_classes = (N+1) * np.ones(self.raw_class_num-N)
        else:
            out_classes = np.setdiff1d(
                self.raw_all_classes,
                self.raw_in_classes)
        return list(out_classes)


    @property
    def new_class_num(self):
        # Convenience
        return np.size(self.raw_in_classes)
    

    @property
    def raw2new_map(self):
        # Users can access raw2new_map but shouldn't set it directly
        return self._raw2new_map
    
    def _set_map(self):
        # Update the raw-to-new mapping array
        ordered_raw_classes = np.concatenate(
            (self.raw_in_classes,
            self.raw_out_classes),
            axis=0
        )

        self._raw2new_map = np.concatenate(
            (np.expand_dims(self.raw_all_classes, axis=1),
            np.expand_dims(ordered_raw_classes, axis=1)),
            axis=1
        )


    def _update_dict(self):
        # TODO:
        # Keep track of the class names using a dict
        # Update the new classes names based on the dict
        print("Not implemented.")


    def set_random_in(self, num=None):
        if num is not None:
            assert num < self.raw_class_num
            self.raw_in_classes = np.random.choice(
                self.raw_class_num, num, replace=False)


    def download(self):
        if self.websites is None or \
            len(self.websites) == 0:
            # Can't download something if we don't know what to download
            raise ValueError

        websites = self.websites
        if isinstance(websites, str):
            websites = [websites]

        for url in websites:
            download_file(url, self.raw_dir, extractor=self.extractor)


    def create_tfrs(self):
        # Create a tensorflow records file for training, validation, 
        # and testing

        # TODO:
        # Current implementation assumes that training, validation, 
        # and testing are all split into seperate files

        def check_and_download(filepaths):
            # Check that the files exist and possibly download them
            # if they do not
            for ff in filepaths:
                flag = not os.path.exists(ff)
                if flag and self.auto_download:
                    self.download()
                    break
                elif flag:
                    raise Exception

        # Create the tfr folder if it doesn't exist
        if not os.path.isdir(self.tfr_dir):
            os.makedirs(self.tfr_dir)

        # Create the training tfr (even if it exists)
        train_raw = self.training_files
        if isinstance(train_raw, str):
            train_raw = [train_raw]
        train_raw = [os.path.join(self.raw_dir, f) for f in train_raw]
        check_and_download(train_raw)
        train_tfr = os.path.join(self.tfr_dir, self.training_tfr)
        self.total_training = self.write_single_tfr(train_raw, train_tfr, 
            retain_all=False) # False to only train on in-distribution

        # Create the validation tfr (even if it exists)
        val_raw = self.validation_files
        if isinstance(val_raw, str):
            val_raw = [val_raw]
        val_raw = [os.path.join(self.raw_dir, f) for f in val_raw]
        check_and_download(val_raw)
        val_tfr = os.path.join(self.tfr_dir, self.validation_tfr)
        self.total_validation = self.write_single_tfr(val_raw, val_tfr, 
            retain_all=True) # True to validate on all data

        # Create the testing tfr (even if it exists)
        test_raw = self.testing_files
        if isinstance(test_raw, str):
            test_raw = [test_raw]
        test_raw = [os.path.join(self.raw_dir, f) for f in test_raw]
        check_and_download(test_raw)
        test_tfr = os.path.join(self.tfr_dir, self.testing_tfr)
        self.total_testing = self.write_single_tfr(test_raw, test_tfr, 
            retain_all=True) # True to test on all data


    def write_single_tfr(self, filepaths_raw, filepath_tfr, retain_all=True):
        # Write one tensorflow record
        # TODO:
        # Explain input arguments

        # TODO:
        # Should we be hard-coding _bytes_feature(...)?

        with tf.python_io.TFRecordWriter(filepath_tfr) as record_writer:
            # Keep track of how many examples are in each dataset
            cnt = 0
            for file in filepaths_raw:
                data, lbls = self.raw_reader(file)
                lbls = map_labels(lbls, self.raw2new_map)
                if not retain_all:
                    # Toss out the data that is out out-of-distribution
                    # Assumes lbls are class numbers in [0,C-1]
                    flag = (lbls < len(self.raw_in_classes))
                    data = data[flag,:]
                    lbls = lbls[flag]

                cnt += len(lbls)
                stream_tfr(data, lbls, record_writer)

        return cnt


    def build_iterators(self, batch_size=64, drop_remainder=False):

        def check_and_build(filepath):
            if not os.path.exists(filepath):
                self.create_tfrs()

        train_map = lambda x: self.augmenter(self.parser(x))
        train_tfr = os.path.join(self.tfr_dir, self.training_tfr)
        check_and_build(train_tfr)
        training_dataset = tf.data.TFRecordDataset(train_tfr)
        training_dataset = training_dataset.map(train_map,
            num_parallel_calls=self.num_parallel_calls)
        training_dataset = training_dataset.batch(batch_size, 
            drop_remainder=drop_remainder)
        training_dataset = training_dataset.shuffle(
            buffer_size=self.buffer_size)

        test_tfr = os.path.join(self.tfr_dir, self.testing_tfr)
        check_and_build(test_tfr)
        testing_dataset = tf.data.TFRecordDataset(test_tfr)
        testing_dataset = testing_dataset.map(self.parser,
            num_parallel_calls=self.num_parallel_calls)
        testing_dataset = testing_dataset.batch(batch_size, 
            drop_remainder=drop_remainder)
        testing_dataset = testing_dataset.shuffle(
            buffer_size=self.buffer_size)

        iterator = tf.data.Iterator.from_structure(
            training_dataset.output_types,
            training_dataset.output_shapes
        )

        if self.total_validation > 0:
            valid_tfr = os.path.join(self.tfr_dir, self.validation_tfr)
            check_and_build(valid_tfr)
            valid_dataset = tf.data.TFRecordDataset(valid_tfr)
            valid_dataset = valid_dataset.map(self.parser,
                num_parallel_calls=self.num_parallel_calls)
            valid_dataset = valid_dataset.batch(batch_size, 
                drop_remainder=drop_remainder)
            valid_dataset = valid_dataset.shuffle(
                buffer_size=self.buffer_size)
        else:
            valid_dataset = None

        return iterator, training_dataset, testing_dataset, valid_dataset


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def stream_tfr(data, lbls, record_writer):
    cnt = 0
    for ii in range(np.shape(data)[0]):
        features = tf.train.Features(feature={
            'image': _bytes_feature(data[ii,:].tobytes()),
            'label': _int64_feature(lbls[ii])
            })
        example = tf.train.Example(features=features)
        record_writer.write(example.SerializeToString())
        cnt += 1

    return cnt


def parser(example, height=32, width=32, depth=32):
    # Default single-example reader
    features = tf.parse_single_example( 
        example, 
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([height * width * depth])
    
    image = tf.cast( 
        tf.transpose( 
            tf.reshape( 
                image, [depth, height, width]), 
            [1,2,0] ), # [0,1,2] ), #
        tf.float32
    ) / 255
    
    # label is currently a scalar with the label number
    # It will be converted using split_labels later
    label = tf.cast(features['label'], tf.int32)
    
    return image, label


def download_file(url, dest_dir, extractor=None):
    # Modified from the tensorflow C10 git repo:
    # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    filename = url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if extractor is not None:
        # A function to extract the data from an archive
        extractor(filepath, dest_dir)


def reader(filepath, bytes_per, dtype='uint8'):
    # Somewhat generic binary reader
    # Assumes data is serialized, [lbl,data]
    data_raw = np.fromfile(filepath, dtype=dtype)
    
    # Organize the data and labels
    data_raw = np.reshape(data_raw, (-1,bytes_per))
    lbls = data_raw[:,0]
    data = data_raw[:,1:]

    return data, lbls


def map_labels(lbls, map):
    
    for cc in range(np.shape(map)[0]):
        flag = (lbls==map[cc,0])
        lbls[flag] = map[cc,1] * np.ones(np.sum(flag), dtype=int)

    return lbls