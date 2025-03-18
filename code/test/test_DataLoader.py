import unittest
import numpy as np
import tensorflow as tf
import torch
import random
from util.torch_to_tf import torch_utils_data_DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        tf.random.set_seed(seed)
        
        # Create example dataset
        self.dataset = [
            {
                "actions": np.random.rand(3), 
                "states": np.random.rand(4), 
                "rewards": np.random.rand(1),
                "next_states": np.random.rand(4), 
                "rgb": np.random.rand(224, 224, 3)
            }
            for _ in range(25)
        ]
        
        # Parameters for dataloaders
        self.batch_size = 5
        # self.shuffle = True
        self.shuffle = False
        
    def test_dataloader_equivalence(self):
        """Test if torch_utils_data_DataLoader produces the same results as torch.utils.data.DataLoader"""
        # Reset seeds before creating dataloaders to ensure same shuffling
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        tf.random.set_seed(seed)
        
        # Create TensorFlow DataLoader
        tf_dataloader = torch_utils_data_DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
        
        # Create PyTorch DataLoader
        torch_dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
        
        # Collect batches from both dataloaders
        tf_batches = []
        for batch in tf_dataloader:
            tf_batches.append(batch)
            
        torch_batches = []
        for batch in torch_dataloader:
            torch_batches.append(batch)
        
        # Assert both dataloaders produce the same number of batches
        self.assertEqual(len(tf_batches), len(torch_batches), 
                         "TensorFlow and PyTorch dataloaders produced different number of batches")
        
        # Compare each batch
        for batch_idx, (tf_batch, torch_batch) in enumerate(zip(tf_batches, torch_batches)):
            # Check if both batches have the same keys
            self.assertEqual(set(tf_batch.keys()), set(torch_batch.keys()),
                            f"Batch {batch_idx} has different keys")
            
            # Check each key-value pair
            for key in tf_batch.keys():
                tf_value = tf_batch[key]
                torch_value = torch_batch[key]
                
                # Check shapes
                self.assertEqual(tf_value.shape, torch_value.shape,
                                f"Shape mismatch for key '{key}' in batch {batch_idx}")
                
                # Convert torch tensor to numpy for comparison
                if isinstance(torch_value, torch.Tensor):
                    torch_value = torch_value.numpy()
                
                # Convert tf tensor to numpy for comparison
                if isinstance(tf_value, tf.Tensor):
                    tf_value = tf_value.numpy()
                
                # Check values
                np.testing.assert_allclose(
                    tf_value, torch_value, 
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"Value mismatch for key '{key}' in batch {batch_idx}"
                )
                
        # print("All batches from TensorFlow and PyTorch dataloaders match!")


if __name__ == "__main__":
    unittest.main()
