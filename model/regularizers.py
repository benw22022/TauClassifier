import tensorflow as tf
import math
from keras import backend
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.util.tf_export import keras_export

class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
  """A regularizer that encourages input vectors to be orthogonal to each other.
  It can be applied to either the rows of a matrix (`mode="rows"`) or its
  columns (`mode="columns"`). When applied to a `Dense` kernel of shape
  `(input_dim, units)`, rows mode will seek to make the feature vectors
  (i.e. the basis of the output space) orthogonal to each other.
  Arguments:
    factor: Float. The regularization factor. The regularization penalty will
      be proportional to `factor` times the mean of the dot products between
      the L2-normalized rows (if `mode="rows"`, or columns if `mode="columns"`)
      of the inputs, excluding the product of each row/column with itself.
      Defaults to 0.01.
    mode: String, one of `{"rows", "columns"}`. Defaults to `"rows"`. In rows
      mode, the regularization effect seeks to make the rows of the input
      orthogonal to each other. In columns mode, it seeks to make the columns
      of the input orthogonal to each other.
  Example:
  >>> regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=0.01)
  >>> layer = tf.keras.layers.Dense(units=4, kernel_regularizer=regularizer)
  """

  def __init__(self, factor=0.01, mode='rows'):
    # _check_penalty_number(factor)
    self.factor = backend.cast_to_floatx(factor)
    if mode not in {'rows', 'columns'}:
      raise ValueError('Invalid value for argument `mode`. Expected one of '
                       f'{{"rows", "columns"}}. Received: mode={mode}')
    self.mode = mode

  def __call__(self, inputs):
    if inputs.shape.rank != 2:
      raise ValueError(
          'Inputs to OrthogonalRegularizer must have rank 2. Received: '
          f'inputs.shape == {inputs.shape}')
    if self.mode == 'rows':
      inputs = tf.math.l2_normalize(inputs, axis=1)
      product = tf.matmul(inputs, tf.transpose(inputs))
      size = inputs.shape[0]
    else:
      inputs = tf.math.l2_normalize(inputs, axis=0)
      product = tf.matmul(tf.transpose(inputs), inputs)
      size = inputs.shape[1]
    product_no_diagonal = product * (1. - tf.eye(size, dtype=inputs.dtype))
    num_pairs = size * (size - 1.) / 2.
    return self.factor * 0.5 * tf.reduce_sum(product_no_diagonal) / num_pairs

  def get_config(self):
    return {'factor': float(self.factor), 'mode': self.mode}