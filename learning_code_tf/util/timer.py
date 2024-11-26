"""
Simple timer from https://github.com/jannerm/diffuser/blob/main/diffuser/utils/timer.py

"""

import time


class Timer:

    def __init__(self):

        print("timer.py: Timer.__init__()")

        self._start = time.time()

    def __call__(self, reset=True):

        print("timer.py: Timer.__call__()")

        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff
