import tensorflow as tf
import numpy as np
import sys

from nets.mobilenet import mobilenet
from preprocessing.mobilenet_preprocessing import preprocess_for_eval
from datasets.imagenet import create_readable_names_for_imagenet_labels

import scipy.misc

slim = tf.contrib.slim

IMAGENET_CLASSES = 1001

with tf.Session() as sess:

    with open('final1.jpg', 'rb') as f:
        image = f.read()

    image = tf.image.decode_jpeg(image, channels=0)
    image = tf.reshape(image, (1825, 2738, 3))
    image = preprocess_for_eval(image, 224, 224)
    image = tf.expand_dims(image, axis=0)

    logits, _ = mobilenet(image, num_classes=IMAGENET_CLASSES)

    softs = tf.nn.softmax(logits)
    top_5 = tf.nn.top_k(softs, k=5)

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    print('restoring weights')
    saver.restore(sess, 'weights/model.ckpt-906808')

    image, softs_out, top_5_out = sess.run([image, softs, top_5])
    scipy.misc.imsave('out.png', image[0,:,:,:])

    values, indices = top_5_out.values, top_5_out.indices

    print('\n')
    print('iii')

    labels_to_names = create_readable_names_for_imagenet_labels()
    for v, i in zip(values[0], indices[0]):
        print('%s with probability %s' % (labels_to_names[i], v))
    
    print('\n')
    print('iv')

    for v in tf.global_variables():
        if not any(str(i) in v.name for i in range(3,17)) or 'global_step' in v.name:
            print(v.name, v.shape)

    print('\n')
    print('v')
    for v in tf.global_variables():
        if 'gamma' in v.name:
            print(v)
        if 'conv_1' in v.name:
            if 'weights' in v.name:
                weights = sess.run(v)
            if 'beta' in v.name:
                beta = sess.run(v)
            if 'moving_mean' in v.name:
                mean = sess.run(v)
            if 'moving_variance' in v.name:
                variance = sess.run(v)
    
    A = (1/np.sqrt(variance + 0.001))

    print('calculated combined weights and biases')
    combined_weights = weights*A
    combined_biases = beta - (A*mean)

    print('\n')
    print('vi')

    print('Don\'t have time to retrain a new final fully connected layer on imagenet examples')
    print('so next best thing is to compare the logits of these classes and return the labels with highest probabilities')

    # indices corresponding to given classes (found by checking labels_to_name)
    indices = range(718, 728)
    print([labels_to_names[ind] for ind in indices])

    softs_out = softs_out[0]
    relevant_logits = softs_out[indices]

    ordered = np.argsort(relevant_logits)[::-1]

    for i in range(5):
        print(labels_to_names[indices[ordered[i]]])
