import json

import matplotlib.pyplot as plt
import scipy.misc
from matplotlib import patches


class SVHNDataset:
    def __init__(self, folder, json_file_name, mean, std):
        self.folder = folder
        file = open(self.folder + json_file_name, "r")
        self.json_file = json.load(file)
        self.index = 0
        self.mean = mean
        self.std = std

    def load(self, count, debug=False):
        examples = []
        labels = []
        positions = []
        end_of_file = False
        for i in range(count):
            try:
                image_object = self.json_file[self.index]
                image_name = image_object["file"]
                example = scipy.misc.imread(self.folder + image_name)
                example = (example - self.mean) / self.std
                examples.append(example)
                labels.append(image_object["label"])
                positions.append(image_object["position"])
                if debug:
                    fig, ax = plt.subplots(1)
                    ax.imshow(example, cmap='gray')
                    x = image_object["position"][0]
                    y = image_object["position"][1]
                    h = image_object["position"][2]
                    w = image_object["position"][3]
                    print(image_object["label"])
                    rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r',
                                             facecolor='none')
                    point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.add_patch(point)
                    plt.show()
                self.index += 1
            except IndexError:
                self.index = 0
                i -= 1
                end_of_file = True
        return {"examples": examples, "labels": labels, "positions": positions, "end_of_file": end_of_file}
