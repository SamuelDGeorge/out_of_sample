
import tensorflow as tf
import numpy as np

def get_image(image_file):
    if not tf.gfile.Exists(image_file):
        print("fail")
    image_data = tf.gfile.FastGFile(image_file, 'rb').read()
    return(image_data)

def get_file_lists(data_dir):
    import glob
 
    train_list = glob.glob(data_dir + '/' + 'train-*')
    valid_list = glob.glob(data_dir + '/' + 'validation-*')
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list
 
def parse_record(raw_record, is_training):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature([], dtype=tf.string, default_value='default.JPEG')
    }
 
    parsed = tf.parse_single_example(raw_record, keys_to_features)
 
    image = tf.image.decode_jpeg(
        tf.reshape(parsed['image/encoded'], shape=[]),
        channels=3)

    # Note that tf.image.convert_image_dtype scales the image data to [0, 1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #image = tf.image.resize_image_with_crop_or_pad(image,331,331)
    
    image = tf.image.resize_images(image, [331,331])

    offset_num = np.array([-1])
    offset_t = tf.constant(offset_num, dtype=tf.int64)
    label = tf.add(parsed['image/class/label'],offset_t)
    label = tf.cast(
        tf.reshape(label, shape=[]),
        dtype=tf.int32)

    filename = tf.reshape(parsed['image/filename'], shape=[])
    
 
    return image, label, filename

def get_batch(is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1):
    dataset = tf.data.TFRecordDataset(filenames)
 
    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)
 
    dataset = dataset.map(lambda value: parse_record(value, is_training),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
 
    features, labels = iterator.get_next()
 
    return features, labels

def build_iterator(is_training, filenames, batch_size, num_epochs=1000, num_parallel_calls=12):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    dataset = dataset.map(lambda value: parse_record(value, is_training),
                            num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator

def build_dataset(is_training, filenames, batch_size, num_epochs=1000, num_parallel_calls=12):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    dataset = dataset.map(lambda value: parse_record(value, is_training),
                            num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset

def get_values_imagenet(sess, image,label,filename):
    im, lab, file = sess.run([image,label,filename])
    return im,lab - 1, file

def get_values_bounded(sess, image,label,filename, offset):
    im, lab, file = sess.run([image,label,filename])
    lab = lab + offset
    return im,lab,file

def get_values_bounded_points(sess, image,p1,p2,p3,p4,filename, offset):
    im,y1,y2,y3,y4,file = sess.run([image,p1,p2,p3,p4,filename])
    y1 = y1 + offset
    y2 = y2 + offset
    y3 = y3 + offset
    y4 = y4 + offset
    return im,y1,y2,y3,y4,file