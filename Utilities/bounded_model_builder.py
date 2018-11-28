
import tensorflow as tf
import numpy as np
 
def parse_bounded_record(raw_record, is_training, num_points,offset):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([num_points], dtype=tf.float32, default_value=np.zeros(num_points)),
        'image/filename':
            tf.FixedLenFeature([], dtype=tf.string, default_value='default.JPEG')
    }
 
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]),
        1)

    # Note that tf.image.convert_image_dtype scales the image data to [0, 1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
 
    image = tf.image.resize_image_with_crop_or_pad(image,331,331)
    image = tf.image.grayscale_to_rgb(image)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[num_points]),
        dtype=tf.float32)
    
    offset_num = np.array([offset,offset,offset,offset,offset,offset,offset,offset])
    offset_t = tf.constant(offset_num, dtype=tf.float32)
    label = tf.add(label,offset_t)

    filename = tf.reshape(parsed['image/filename'], shape=[])
    
 
    return image, label, filename

def parse_bounded_record_points(raw_record, is_training, num_points,offset):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/p1':
            tf.FixedLenFeature([2], dtype=tf.float32, default_value=np.zeros(2)),
        'image/class/p2':
            tf.FixedLenFeature([2], dtype=tf.float32, default_value=np.zeros(2)),
        'image/class/p3':
            tf.FixedLenFeature([2], dtype=tf.float32, default_value=np.zeros(2)),
        'image/class/p4':
            tf.FixedLenFeature([2], dtype=tf.float32, default_value=np.zeros(2)),    
        'image/filename':
            tf.FixedLenFeature([], dtype=tf.string, default_value='default.JPEG')
    }
 
    parsed = tf.parse_single_example(raw_record, keys_to_features)
    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]),
        1)

    # Note that tf.image.convert_image_dtype scales the image data to [0, 1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
 
    image = tf.image.resize_image_with_crop_or_pad(image,331,331)
    image = tf.image.grayscale_to_rgb(image)

    offset_num = np.array([offset,offset])
    offset_t = tf.constant(offset_num, dtype=tf.float32)

    p1 = tf.cast(
        tf.reshape(parsed['image/class/p1'], shape=[2]),
        dtype=tf.float32)
    
    p1 = tf.add(p1,offset_t)

    p2 = tf.cast(
        tf.reshape(parsed['image/class/p2'], shape=[2]),
        dtype=tf.float32)
    
    p2 = tf.add(p2,offset_t)

    p3 = tf.cast(
        tf.reshape(parsed['image/class/p3'], shape=[2]),
        dtype=tf.float32)
    
    p3 = tf.add(p3,offset_t)

    p4 = tf.cast(
        tf.reshape(parsed['image/class/p4'], shape=[2]),
        dtype=tf.float32)
    
    p4 = tf.add(p4,offset_t)

    filename = tf.reshape(parsed['image/filename'], shape=[])
    
 
    return image,p1,p2,p3,p4,filename

def get_bounded_batch(is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1, num_points=8, offset=37.5):
    dataset = tf.data.TFRecordDataset(filenames)
 
    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)
 
    dataset = dataset.map(lambda value: parse_bounded_record(value, is_training, num_points,offset),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
 
    features, labels = iterator.get_next()
 
    return features, labels

def build_bounded_iterator(is_training, filenames, batch_size, num_epochs=1000, num_parallel_calls=12, num_points = 8, offset=37.5):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    dataset = dataset.map(lambda value: parse_bounded_record(value, is_training, num_points, offset),
                            num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator

def build_bounded_iterator_points(is_training, filenames, batch_size, num_epochs=1000, num_parallel_calls=12, num_points = 8, offset=37.5):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    dataset = dataset.map(lambda value: parse_bounded_record_points(value, is_training, num_points, offset),
                            num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator
