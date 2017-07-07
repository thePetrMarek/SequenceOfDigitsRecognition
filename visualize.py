import os

import matplotlib.pyplot as plt


class Visualize:
    def visualize(self, image, label, folder):
        # TODO check if folder already exists
        try:
            os.mkdir(folder)
        except:
            pass
        plt.imshow(image, cmap='gray')
        plt.title(label)
        # TODO check if file already exists
        plt.savefig(os.path.join(folder, str(label) + ".png"))
