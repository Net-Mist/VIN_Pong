"""Builds the VIN network.
structure from the tensorflow example :
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

Implements the tensorflow inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.
"""

import tensorflow as tf
import math


def placeholders_openai(number_block: int) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """
    :return: the placeholders
    """
    y_player_placeholder = tf.placeholder(tf.float32,
                                          shape=(1, number_block),
                                          name='player_position_placeholder')
    other_information_placeholder = tf.placeholder(tf.float32,
                                                   shape=(1, 5),
                                                   name='other_information_placeholder')
    return y_player_placeholder, other_information_placeholder


def placeholders_training(batch_size: int, number_block: int) -> (
        tf.Tensor, tf.Tensor, tf.Tensor):
    """
    :param batch_size: number of images in the batch
    :param number_block: number of blocks
    :return: the placeholders
    """
    y_player_placeholder = tf.placeholder(tf.float32,
                                          shape=(batch_size, number_block),
                                          name='player_position_placeholder')
    other_information_placeholder = tf.placeholder(tf.float32,
                                                   shape=(batch_size, 5),
                                                   name='other_information_placeholder')

    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size, name='Input_label')

    return y_player_placeholder, other_information_placeholder, labels_placeholder


def inference(y_player_placeholder: tf.placeholder, other_information_placeholder: tf.placeholder, num_action: int,
              k_rec: int, number_block: int) -> tf.Tensor:
    """
    Build the VIN model up to where it may be used for inference.
    The model is built according to the original VIN paper

    :param y_player_placeholder:
    :param other_information_placeholder:
    :param num_action: number of possible actions
    :param k_rec: number of iteration inside the VIN module
    :param number_block:
    :return: tensor with the computed logits
    """

    # First part : fully-connected layer to calcul R
    with tf.name_scope('prepare_VIN'):
        with tf.name_scope('hidden_layer_1'):
            weights = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 150], stddev=0.1), name='weights')
            biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=[150], dtype=tf.float32),
                                 name='biases')
            hidden = tf.nn.relu(tf.matmul(other_information_placeholder, weights) + biases)
        with tf.name_scope('hidden_layer_2'):
            weights = tf.Variable(initial_value=tf.truncated_normal(shape=[150, 100], stddev=0.1), name='weights')
            biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=[100], dtype=tf.float32),
                                 name='biases')
            hidden = tf.nn.relu(tf.matmul(hidden, weights) + biases)
        with tf.name_scope('hidden_layer_3'):
            weights = tf.Variable(initial_value=tf.truncated_normal(shape=[100, number_block], stddev=0.1),
                                  name='weights')
            biases = tf.Variable(initial_value=tf.constant(value=0.1, shape=[number_block], dtype=tf.float32),
                                 name='biases')
            r = tf.nn.relu(tf.matmul(hidden, weights) + biases)

    # Value iteration part
    v = tf.fill(tf.shape(r), 0.0)
    v = tf.expand_dims(v, 2)
    r = tf.expand_dims(r, 2)
    with tf.variable_scope('vi') as scope:
        for irec in range(k_rec):
            with tf.name_scope('iter%d' % irec):
                if irec == 1:
                    scope.reuse_variables()
                # concatenate V with R
                v_concat = tf.concat_v2([v, r], 2)

                filters = tf.get_variable('weights', [3, 2, num_action],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                          dtype=tf.float32)
                conv = tf.nn.conv1d(v_concat, filters, 1, padding='SAME')
                biases = tf.get_variable('bias', [num_action], initializer=tf.constant_initializer(0.0))
                q = tf.nn.bias_add(conv, biases, name="Q")
                # activation_summary(Q)
                v = tf.reduce_max(q, reduction_indices=[2], keep_dims=True,
                                  name="V")  # TODO : reduction_indices is deprecated, use axis instead

    # attention part
    with tf.name_scope('attention'):
        Qa_img = tf.mul(q, tf.tile(tf.expand_dims(y_player_placeholder, 2), [1, 1, num_action]), name='Qa_img')
        Qa = tf.reduce_sum(Qa_img, [1], name="Qa")

    # reactive policy (dense layer with softmax?)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=[num_action, num_action],
                                                                stddev=1.0 / math.sqrt(float(num_action))),
                              name='weights')
        biases = tf.get_variable('b_policy', [num_action], initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(Qa, weights) + biases
        softact = tf.nn.softmax(logits, name='softact')

    return logits


def classification_loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Create the gradient descent optimizer with the given learning rate.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # from the website TODO: tester l'autre
    # optimizer = tf.train.RMSPropOptimizer(learning_rate * (tf.pow(0.9, (global_step / 1000))),
    #                                       decay=0.9)  # from VIN implementation
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1e-6, centered=True)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
