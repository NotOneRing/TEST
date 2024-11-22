import pytest
import tensorflow as tf

from sampling import extract

# Test case 1: Basic functionality
def test_extract_basic():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = tf.constant([0, 1, 2])
    x_shape = (3, 3)
    expected_output = tf.constant([[1], [5], [9]])
    actual_output = extract(a, t, x_shape)
    assert tf.reduce_all(tf.equal(actual_output, expected_output))

# Test case 2: Different t values
def test_extract_different_t():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = tf.constant([1, 0, 2])
    x_shape = (3, 3)
    expected_output = tf.constant([[2], [1], [9]])
    actual_output = extract(a, t, x_shape)
    assert tf.reduce_all(tf.equal(actual_output, expected_output))

# Test case 3: Different x_shape
def test_extract_different_x_shape():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = tf.constant([0, 1, 2])
    x_shape = (3, 2)
    expected_output = tf.constant([[1, 2], [5, 6], [9, 0]])
    actual_output = extract(a, t, x_shape)
    assert tf.reduce_all(tf.equal(actual_output, expected_output))

# Test case 4: Empty t
def test_extract_empty_t():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = tf.constant([])
    x_shape = (3, 3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        extract(a, t, x_shape)

# Test case 5: Invalid t values
def test_extract_invalid_t():
    a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t = tf.constant([-1, 3])
    x_shape = (3, 3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        extract(a, t, x_shape)


test_extract_basic()
