"""
Normalization for Furniture-Bench environments.

TODO: use this normalizer for all benchmarks.

"""

import torch.nn as nn


class LinearNormalizer(nn.Module):
    def __init__(self):

        print("furniture_normalizer.py: LinearNormalizer.__init__()", flush = True)

        super().__init__()
        self.stats = nn.ParameterDict()

    def fit(self, data_dict):

        print("furniture_normalizer.py: LinearNormalizer.fit()", flush = True)

        for key, tensor in data_dict.items():
            min_value = tensor.min(dim=0)[0]
            max_value = tensor.max(dim=0)[0]

            # Check if any column has only one value throughout
            diff = max_value - min_value
            constant_columns = diff == 0

            # Set a small range for constant columns to avoid division by zero
            min_value[constant_columns] -= 1
            max_value[constant_columns] += 1

            self.stats[key] = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_value, requires_grad=False),
                    "max": nn.Parameter(max_value, requires_grad=False),
                },
            )
        self._turn_off_gradients()

    def _normalize(self, x, key):

        print("furniture_normalizer.py: LinearNormalizer._normalize()", flush = True)

        stats = self.stats[key]
        x = (x - stats["min"]) / (stats["max"] - stats["min"])
        x = 2 * x - 1
        return x

    def _denormalize(self, x, key):

        print("furniture_normalizer.py: LinearNormalizer._denormalize()", flush = True)

        stats = self.stats[key]
        x = (x + 1) / 2
        x = x * (stats["max"] - stats["min"]) + stats["min"]
        return x

    def forward(self, x, key, forward=True):

        print("furniture_normalizer.py: LinearNormalizer.forward()", flush = True)
 
        if forward:
            return self._normalize(x, key)
        else:
            return self._denormalize(x, key)

    def _turn_off_gradients(self):

        print("furniture_normalizer.py: LinearNormalizer._turn_off_gradients()", flush = True)

        for key in self.stats.keys():
            for stat in self.stats[key].keys():
                self.stats[key][stat].requires_grad = False

    def load_state_dict(self, state_dict):

        print("furniture_normalizer.py: LinearNormalizer.load_state_dict()", flush = True)

        stats = nn.ParameterDict()
        for key, value in state_dict.items():
            if key.startswith("stats."):
                param_key = key[6:]
                keys = param_key.split(".")
                current_dict = stats
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = nn.ParameterDict()
                    current_dict = current_dict[k]
                current_dict[keys[-1]] = nn.Parameter(value)

        self.stats = stats
        self._turn_off_gradients()

        return f"<Added keys {self.stats.keys()} to the normalizer.>"

    def keys(self):

        print("furniture_normalizer.py: LinearNormalizer.keys()", flush = True)

        return self.stats.keys()
