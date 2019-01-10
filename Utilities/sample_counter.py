import tensorflow as tf

def get_likelihood_avg(class_labels, number_of_classes, likelihood_list, likelihood_averages, number_of_batchs):
    mean_array = []
    for i in range(0,number_of_classes):
        class_select = tf.where(tf.equal(class_labels,i))
        like_class = tf.gather_nd(likelihood_list,class_select)
        sample_number = tf.cast(tf.size(like_class), tf.float32)      
        mean = tf.reduce_mean(like_class)
        current_mean = tf.gather(likelihood_averages, i)
        batch_count = tf.gather(number_of_batchs, i)
        new_mean = ((current_mean * batch_count) + (sample_number * mean))/(batch_count + sample_number) 
        
        to_add = tf.cond(tf.equal(sample_number,0), lambda: current_mean, lambda: new_mean)  
        mean_array.append(to_add)
    mean_array = tf.stack(mean_array)
        
    return mean_array

def get_likelihood_stdev(class_labels, number_of_classes, likelihood_list, likelihood_std, number_of_batchs):
    stdev_array = []
    for i in range(0,number_of_classes):
        class_select = tf.where(tf.equal(class_labels,i))
        like_class = tf.gather_nd(likelihood_list,class_select)
        sample_number = tf.cast(tf.size(like_class), tf.float32)
        mean = tf.reduce_mean(like_class)
        squares = tf.square(tf.subtract(like_class, mean))
        stdev = tf.sqrt(tf.reduce_mean(squares))      
        current_stdev = tf.gather(likelihood_std, i)
        batch_count = tf.gather(number_of_batchs, i)
        new_std = ((current_stdev * batch_count) + (sample_number * stdev))/(batch_count + sample_number) 
        to_add = tf.cond(tf.equal(sample_number,0), lambda: current_stdev, lambda: new_std)  
        stdev_array.append(to_add)
    stdev_array = tf.stack(stdev_array)
        
    return stdev_array

def get_sample_number(class_labels, number_of_classes, likelihood_list, number_of_batchs):
    sample_array = []
    for i in range(0,number_of_classes):
        class_select = tf.where(tf.equal(class_labels,i))
        like_class = tf.gather_nd(likelihood_list,class_select)
        sample_number = tf.cast(tf.size(like_class),tf.float32)
        batch_count = tf.gather(number_of_batchs, i)
        samples = sample_number + batch_count
        sample_array.append(samples)
    sample_array = tf.stack(sample_array)
        
    return sample_array