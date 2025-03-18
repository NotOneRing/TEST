import tensorflow as tf
import unittest
import numpy as np
import os

from util.torch_to_tf import nn_Linear, nn_ReLU

from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
class C(tf.keras.Model):
    def __init__(self, **kwargs):
        super(C, self).__init__()
        self.dense_c = nn_Linear(8, 4)
        # tf.keras.layers.Dense(4, activation='relu')
        self.relu = nn_ReLU()
    
    def call(self, inputs):
        x = self.dense_c(inputs)
        return self.relu(x)


@register_keras_serializable(package="Custom")
class B(tf.keras.Model):
    def __init__(self, **kwargs):
        super(B, self).__init__()
        self.dense_b = nn_Linear(16, 8)
        # tf.keras.layers.Dense(8, activation='relu')
        self.relu = nn_ReLU()
        self.c = C()
    
    def call(self, inputs):
        x = self.dense_b(inputs)
        x = self.relu(x)
        return self.c(x)


@register_keras_serializable(package="Custom")
class A(tf.keras.Model):
    def __init__(self, **kwargs):
        super(A, self).__init__()
        self.dense_a = nn_Linear(10, 16)
        # tf.keras.layers.Dense(16, activation='relu')
        self.relu = nn_ReLU()
        self.b = B()
    
    def call(self, inputs):
        x = self.dense_a(inputs)
        x = self.relu(x)
        return self.b(x)


class TestSaveLoadModels(unittest.TestCase):
    def setUp(self):
        # Create random data for testing
        self.x_train = tf.random.normal((32, 10))  # input shape：32 samples，every sample of 10 dimension
        self.y_train = tf.random.normal((32, 4))   # output shape：32 samples，every sample of 4 dimension
        
        # Model file path
        self.model_path = "nested_model1.keras"
    
    def tearDown(self):
        # Clean up - remove the saved model file after tests
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
    
    def test_save_load_model(self):
        """Test saving and loading nested models with weights preservation"""
        # create model instance
        model_a = A()

        # # compile and train
        # model_a.compile(optimizer='adam', loss='mse')
        # model_a.fit(self.x_train, self.y_train, epochs=3, verbose=0)
        pred = model_a(self.x_train)
        
        # save model (save all contents of B and C recursively)
        model_a.save(self.model_path)

        from tensorflow.keras.utils import get_custom_objects
        cur_dict = {
            'A': A,
            'B': B,
            'C': C,  
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
        }
        get_custom_objects().update(cur_dict)
        loaded_model_a = tf.keras.models.load_model(self.model_path ,  custom_objects=get_custom_objects())
        
        # # load models
        # loaded_model_a = tf.keras.models.load_model(self.model_path)

        # check if weights are retained
        outputs_original = model_a(self.x_train)
        outputs_loaded = loaded_model_a(self.x_train)

        # Assert outputs are close
        self.assertTrue(np.allclose(outputs_original.numpy(), outputs_loaded.numpy()))
        
        # Check difference is close to 0
        diff = tf.reduce_sum(tf.abs(outputs_original - outputs_loaded))
        self.assertLess(diff, 1e-5)
        
        # Check optimizer configuration is preserved
        self.assertEqual(loaded_model_a.optimizer.__class__.__name__, 'Adam')


if __name__ == '__main__':
    unittest.main()

# test = TestSaveLoadModels()
# test.setUp()  # 1. run setUp() to create testing data
# test.test_save_load_model()  # 2. running test
# test.tearDown()  # 3. use tearDown() to clean environment




