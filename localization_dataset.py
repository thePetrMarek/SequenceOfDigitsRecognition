import pickle
import matplotlib.pyplot as plt
from matplotlib import patches


class LocalizationDataset:
    def __init__(self, file_name):
        self.file = open(file_name, 'rb')
        self.file_name = file_name

    def load(self, count, debug=False):
        examples = []
        labels = []
        positions = []
        end_of_file = False
        for i in range(count):
            try:
                object = pickle.load(self.file)
                examples.append(object["example"])
                labels.append(object["label"])
                positions.append(object["position"])
                if debug:
                    fig, ax = plt.subplots(1)
                    ax.imshow(object["example"], cmap='gray')
                    x = object["position"][0]
                    y = object["position"][1]
                    h = object["position"][2]
                    w = object["position"][3]
                    print(object["label"])
                    rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r',
                                             facecolor='none')
                    point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.add_patch(point)
                    plt.show()
            except EOFError:
                self.file = open(self.file_name, 'rb')
                i -= 1
                end_of_file = True
        return {"examples": examples, "labels": labels, "positions": positions, "end_of_file": end_of_file}
