import tensorflow as tf
from PIL import Image as PI
import numpy as np

def generate_image_array(filename, height, width, scale_to_one = True):
    image = PI.open(filename)
    image = image.resize((height, width), PI.ANTIALIAS)
    to_return = np.array(image).reshape(1,299,299,3)
    if scale_to_one:
        return to_return/255
    return to_return

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

class Binary_Categorical_Predictor():
    def __init__(self, model_location, label_array):
        self.model = model_location
        self.labels = label_array
    def print_image_probability(self, input_image_location):
        with tf.Session() as sess:
            #restore graph from meta and restore variables
            new_saver = tf.train.import_meta_graph(self.model + '.meta')
            new_saver.restore(sess, self.model)
            soft = tf.get_default_graph().get_tensor_by_name("DNN_Classifier/Softmax:0")
            input_tensor = tf.get_default_graph().get_tensor_by_name("input:0")
            image = generate_image_array(input_image_location, 299, 299)
            val = soft.eval(feed_dict={input_tensor: image})
        print("Probability of " + self.labels[0] + ": " + str(val[0][0]))
        print("Probability of " + self.labels[1] + ": " + str(val[0][1]))