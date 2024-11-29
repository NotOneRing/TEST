# from agent.pretrain.train_agent import EMA

import pytest
from agent.pretrain.train_agent import EMA

from omegaconf import OmegaConf

import tensorflow as tf

import os

import hydra

@hydra.main(
    version_base=None,
    config_path=os.path.join(
        os.getcwd(), "cfg"
    ),  # possibly overwritten by --config-path
)
# def main(cfg: OmegaConf):

#     print("run.py: main()")

    # resolve immediately so all the ${now:} resolvers will use the same time.



OmegaConf.resolve(cfg)


# Test case 1: Normal update
def test_update_model_average_normal():
    cfg = OmegaConf.create({'decay': 0.9})
    ema = EMA(cfg)

    ma_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    current_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

    # Set some initial values for the weights
    ma_model.layers[0].set_weights([np.random.randn(10, 10)])
    current_model.layers[0].set_weights([np.random.randn(10, 10)])

    ema.update_model_average(ma_model, current_model)

    # Assert that the weights have been updated
    assert not np.array_equal(ma_model.layers[0].get_weights(), current_model.layers[0].get_weights())

# Test case 2: Zero decay
def test_update_model_average_zero_decay():
    cfg = OmegaConf.create({'decay': 0})
    ema = EMA(cfg)

    ma_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    current_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

    # Set some initial values for the weights
    ma_model.layers[0].set_weights([np.random.randn(10, 10)])
    current_model.layers[0].set_weights([np.random.randn(10, 10)])

    ema.update_model_average(ma_model, current_model)

    # Assert that the weights have not been updated
    assert np.array_equal(ma_model.layers[0].get_weights(), current_model.layers[0].get_weights())

# Test case 3: Different layer structures
def test_update_model_average_different_layers():
    cfg = OmegaConf.create({'decay': 0.9})
    ema = EMA(cfg)

    ma_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    current_model = tf.keras.models.Sequential([tf.keras.layers.Dense(5)])

    # Set some initial values for the weights
    ma_model.layers[0].set_weights([np.random.randn(10, 10)])
    current_model.layers[0].set_weights([np.random.randn(5, 5)])

    with pytest.raises(AssertionError):
        ema.update_model_average(ma_model, current_model)
