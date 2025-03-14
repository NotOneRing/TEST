import unittest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

class TestTensorFlow(unittest.TestCase):
    """Test class for TensorFlow functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Load pre-trained VGG16 model with ImageNet weights, excluding top layers
        self.model_1 = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
        self.model_1.trainable = False
        
        # Extract layers from the model
        self.layers = [l for l in self.model_1.layers]
        
        # Reset outbound nodes for each layer
        for layer in self.layers:
            layer._outbound_nodes = []
    
    def test_model_creation(self):
        """Test creating a new model from VGG16 layers."""
        # Create input layer with shape (224, 224, 3)
        input = keras.Input(shape=(224, 224, 3))
        
        # Connect layers sequentially
        x = self.layers[1](input)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = self.layers[7](x)
        x = self.layers[8](x)
        x = self.layers[9](x)
        x = self.layers[10](x)
        x = self.layers[11](x)
        x = self.layers[12](x)
        x = self.layers[13](x)
        
        # Add a Dense layer with 38 outputs and softmax activation
        out = Dense(38, activation='softmax')(x)
        
        # Create the final model
        result_model = tf.keras.Model(inputs=input, outputs=out)
        
        # Verify model structure
        self.assertEqual(len(result_model.layers), 15)  # 13 VGG layers + input + dense
        self.assertEqual(result_model.output_shape[-1], 38)  # Output dimension should be 38
        self.assertFalse(result_model.layers[1].trainable)  # Layers should be non-trainable
    
    def test_model_summary(self):
        """Test model summary generation."""
        # Print model summary (this is just for information, not actual testing)
        self.model_1.summary()
        
        # Create a simple model for testing summary
        input = keras.Input(shape=(224, 224, 3))
        x = self.layers[1](input)
        out = Dense(38, activation='softmax')(x)
        result_model = tf.keras.Model(inputs=input, outputs=out)
        
        # Verify that summary can be generated without errors
        summary = []
        result_model.summary(print_fn=lambda x: summary.append(x))
        
        # Check that summary contains expected information
        summary_text = '\n'.join(summary)
        print("summary_text = ", summary_text)
        self.assertIn('Model: "', summary_text)
        self.assertIn('Dense', summary_text)
        self.assertIn('38', summary_text)  # Output dimension


if __name__ == '__main__':
    unittest.main()
