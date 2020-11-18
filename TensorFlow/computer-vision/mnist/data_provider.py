import collections
import logging
import tensorflow_datasets as tfds

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataProvider(object):

    tf_log = logging.getLogger('tf_log')

    def __init__(self, name=None):
        if name:
            self.builder = tfds.builder(name)
            self.builder.download_and_prepare()

    def getData(self, shuffle_files=False):
        train = self.builder.as_dataset(split=tfds.Split.TRAIN, shuffle_files=shuffle_files)
        test = self.builder.as_dataset(split=tfds.Split.TEST, shuffle_files=shuffle_files)
        self.tf_log.info(f"Dataset: {self.builder.info.name}")
        return Datasets(train=train, test=test, validation=None), self.builder.info
