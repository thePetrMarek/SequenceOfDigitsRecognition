import os

import matplotlib.pyplot as plt


class Visualize:
    def visualize(self, image, label, folder):
        self.visualize_with_correct(image, label, None, folder)

    def visualize_with_correct(self, image, recognized_label, correct_label, folder):
        # TODO check if folder already exists
        try:
            os.mkdir(folder)
        except:
            pass
        plt.imshow(image, cmap='gray')
        if correct_label is not None:
            plt.title("Recognized: " + str(recognized_label) + "\n" + "True: " + str(correct_label))
        else:
            plt.title(str(recognized_label))
        # TODO check if file already exists
        plt.savefig(os.path.join(folder, str(correct_label) + ".png"))
