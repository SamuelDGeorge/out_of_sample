import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def make_and_push_instance(X_Data, y_Data, instance_queue, batch_size):
    features, target = random_batch(X_Data, y_Data, batch_size)
    enqueue_instance = instance_queue.enqueue([features, target])
    return enqueue_instance

def log_dir_build(root_dir, prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = root_dir
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

class DNN_Model:
    def __init__(self, n_epochs=200, batch_size=10, n_outputs=4, log_dir="./tf_logs", model_name="dnn_model.ckpt", device_dictionary={'GPU':0}):
        self.n_epochs=n_epochs
        self.batch_size=batch_size
        self.n_outputs=n_outputs
        self.log_dir=log_dir
        self.model_name=model_name
        self.model_path=""
        self.device_dictionary=device_dictionary

    def fit(self, X_data, y_data):
        #Build the graph
        reset_graph()
        n_inputs = len(X_data[0])
        num_batches = len(X_data) // self.batch_size
        n_hidden1 = n_inputs
        n_hidden2 = 150
        n_hidden3 = 100
        n_hidden4 = 50
        dropout_rate = 0.5


        q = tf.FIFOQueue(capacity=num_batches * self.n_epochs, dtypes=[tf.float32, tf.int64], 
                 shapes=[(self.batch_size, n_inputs), (self.batch_size)], 
                        name="data", shared_name="shared_data")

        #build the 12 functions to be run in parralel
        add_to_queue = [make_and_push_instance(X_data, y_data, q, self.batch_size) for i in range(12)]
        queue_runner = tf.train.QueueRunner(q, add_to_queue)

        X_out, y_out = q.dequeue()

        X = tf.placeholder(tf.float32, shape = (None, n_inputs), name="X_input")
        y = tf.placeholder(tf.int64, shape=(None), name = "y_input")

        training = tf.placeholder_with_default(False, shape=(), name = 'training')
        bn_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)
        he_init = tf.contrib.layers.variance_scaling_initializer()

        #using tensorflow
        with tf.name_scope("dnn"):
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", kernel_initializer=he_init)
            hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
            bn1 = bn_batch_norm_layer(hidden1_drop)
            bn1_act = tf.nn.elu(bn1)

            hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2", kernel_initializer=he_init)
            hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
            bn2 = bn_batch_norm_layer(hidden2_drop)
            bn2_act = tf.nn.elu(bn2)

            hidden3 = tf.layers.dense(bn2_act, n_hidden3, name="hidden3", kernel_initializer=he_init)
            hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
            bn3 = bn_batch_norm_layer(hidden3_drop)
            bn3_act = tf.nn.elu(bn3)
            
            hidden4 = tf.layers.dense(bn3_act, n_hidden4, name="hidden4", kernel_initializer=he_init)
            hidden4_drop = tf.layers.dropout(hidden4, dropout_rate, training=training)
            bn4 = bn_batch_norm_layer(hidden4_drop)
            bn4_act = tf.nn.elu(bn4)

            logits_before_bn = tf.layers.dense(bn4_act, self.n_outputs, name="outputs")
            logits = bn_batch_norm_layer(logits_before_bn, name="logits")
        #Calculate the loss out fo the last layer

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")

        #Describe a way to train

        learning_rate = 0.01
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)

         #Evaluate the model
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            accuracy_summary = tf.summary.scalar('accuracy', accuracy)


        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        #set up logging
        self.model_path = log_dir_build(self.log_dir, "dnn_out")
        file_writer = tf.summary.FileWriter(self.model_path, tf.get_default_graph())
        
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        config = tf.ConfigProto(device_count = self.device_dictionary)
        config.allow_soft_placement=True
        with tf.Session(config=config) as sess:
            current_best = 0
            init.run()

            #build a gigantic queue for all the data
            coord = tf.train.Coordinator()
            enqueue_threads = queue_runner.create_threads(sess, coord=coord, start=True)

            for epoch in range(self.n_epochs):
                for item in range(num_batches):   
                    X_item, y_item = sess.run([X_out, y_out])

                    sess.run([training_op, extra_update_ops],feed_dict={training: True, X: X_item, y: y_item})
                acc_train = accuracy.eval(feed_dict={X: X_item, y: y_item})
                acc_val, acc_sum = sess.run([accuracy, accuracy_summary],feed_dict={X: X_data, y: y_data})
                file_writer.add_summary(acc_sum, epoch)
                if (acc_val > current_best):
                    current_best = acc_val
                    saver.save(sess, self.model_path + self.model_name)

            coord.request_stop()

    
    def predict(self, X):
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, self.model_path + self.model_name)
            X_new_scaled = X
            logits = tf.get_default_graph().get_tensor_by_name("dnn/logits/batchnorm/add_1:0")
            X_tensor = tf.get_default_graph().get_tensor_by_name("X_input:0")
            training = tf.get_default_graph().get_tensor_by_name("training:0")
            y_raw = logits.eval(feed_dict={training: False, X_tensor: X_new_scaled})
            y_pred = np.argmax(y_raw, axis=1)
            return(y_pred)

def cross_val_score_dnn(model, X_data, y_data, cv=10):
    skfolds = StratifiedKFold(n_splits=cv, random_state=42)
    scores = []
    for train_index, test_index in skfolds.split(X_data, y_data):
        X_train_folds = X_data[train_index]
        y_train_folds = y_data[train_index]
        X_test_folds = X_data[test_index]
        y_test_folds = y_data[test_index]

        model.fit(X_train_folds, y_train_folds)
        y_pred = model.predict(X_test_folds)
        score = f1_score(y_test_folds, y_pred, average='macro')
        scores.append(score)
    return(scores)