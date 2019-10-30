import os
import numpy as np

class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path, mode='train'):
        self.name = name + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path
        self.mode = mode
        self.annotations = self.load_annotations()

    def get_annotations(self):
        return self.annotations

    def load_annotations(self):

        annotation_file = os.path.join(self.data_path, 'imglists', self.image_set + '.txt')
        assert os.path.exists(annotation_file), 'annotations not found at {}'.format(annotation_file)
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()

        return annotations