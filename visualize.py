import os

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


class Visualize:
    def visualize(self, image, label, folder):
        self.visualize_with_correct_label(image, label, None, folder)

    def visualize_with_correct_label(self, image, recognized_label, correct_label, folder):
        self.visualize_with_correct_label_position(image, recognized_label, correct_label, None, None, folder)

    def visualize_with_correct_label_position(self, image, recognized_label, correct_label, predicted_position,
                                              correct_position, folder):
        # TODO check if folder already exists
        try:
            os.makedirs(folder)
        except:
            pass
        fig, ax = plt.subplots(1)
        ax.imshow(np.around(image))

        title = ""
        if correct_label is not None:
            title += "Recognized: " + str(recognized_label) + "\n" + "True: " + str(correct_label) + "\n"
        else:
            title += str(recognized_label) + "\n"
        plt.title(title)

        if predicted_position is not None:
            title += str("Predicted position: Red \n")
            x = predicted_position[0]
            y = predicted_position[1]
            h = predicted_position[2]
            w = predicted_position[3]
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none')
            point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(point)
        if correct_position is not None:
            title += str("True position: Green \n")
            x = correct_position[0]
            y = correct_position[1]
            h = correct_position[2]
            w = correct_position[3]
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='g', facecolor='none')
            point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(point)

        # TODO check if file already exists
        plt.savefig(os.path.join(folder, str(correct_label) + ".png"))

    def visualize_inference(self, image, label, predicted_position):
        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gray')
        title = label
        x = predicted_position[0]
        y = predicted_position[1]
        h = predicted_position[2]
        w = predicted_position[3]
        rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none')
        point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(point)
        plt.title(title)
        plt.show()
