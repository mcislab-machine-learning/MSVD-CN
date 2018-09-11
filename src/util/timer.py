import os
import sys
import time


class Timer:
    def __init__(self):
        self.start_time = 0.
        self.total_time = 0.
        self.steps = []

    @staticmethod
    def __get_time():
        return time.time()

    def start(self):
        self.start_time = self.__get_time()

    def step(self, name):
        time_elapsed = self.__get_time() - self.start_time
        self.steps.append((name, time_elapsed))
        self.total_time += time_elapsed

    def print(self):
        print('total: {:.4f}, {}'.format(self.total_time, ', '.join(
            '{} {:.4f} {:.2f}%'.format(name, time, time * 100.0 / self.total_time) for name, time in self.steps)))