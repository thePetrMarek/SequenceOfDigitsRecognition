import h5py
import scipy.misc
import tqdm
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches


def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def prepare_one_dataset(folder, file, out_file, debug=False):
    mat_data = h5py.File(folder + file)
    size = mat_data['/digitStruct/name'].size

    dataset = []

    max_length = 0
    for _i in tqdm.tqdm(range(size)):
        pic = get_name(_i, mat_data)
        box = get_box_data(_i, mat_data)

        length = len(box["label"])
        if length > max_length:
            max_length = length

        label = []
        for i in range(6):
            if i < length:
                number = box["label"][i]
                if number == 10:
                    number = 0
            else:
                number = 10
            zeros = [0] * 11
            zeros[int(number)] = 1
            label.append(zeros)

        min_left = min(box["left"])
        max_lefts = [x + y for x, y in zip(box["left"], box["width"])]
        max_left = max(max_lefts)

        min_top = min(box["top"])
        max_tops = [x + y for x, y in zip(box["top"], box["height"])]
        max_top = max(max_tops)

        x = ((max_left - min_left) / 2) + min_left
        y = ((max_top - min_top) / 2) + min_top

        w = max_left - min_left
        h = max_top - min_top

        old_picture = scipy.misc.imread(folder + pic)

        if debug:
            fig, ax = plt.subplots(1)
            ax.imshow(old_picture, cmap='gray')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none')
            point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(point)
            plt.show()
            print()

        im = Image.open(folder + pic)
        original_size = im.size
        new_im = im.resize((256, 128), Image.ANTIALIAS)
        new_pic = folder + "new/" + pic
        if not os.path.exists(folder + "new/"):
            os.makedirs(folder + "new/")
        new_im.save(new_pic)

        height_ratio = 128 / original_size[1]
        width_ratio = 256 / original_size[0]

        x = (((max_left - min_left) / 2) + min_left) * width_ratio
        y = (((max_top - min_top) / 2) + min_top) * height_ratio

        w = (max_left - min_left) * width_ratio
        h = (max_top - min_top) * height_ratio
        position = [x, y, h, w]

        example = {"file": pic, "label": label, "position": position}

        new_example = scipy.misc.imread(new_pic)

        if debug:
            print(label)
            print(position)
            fig, ax = plt.subplots(1)
            ax.imshow(new_example, cmap='gray')
            rect = patches.Rectangle((x - (w / 2), y - (h / 2)), w, h, linewidth=1, edgecolor='r', facecolor='none')
            point = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(point)
            plt.show()
            print()

        dataset.append(example)
    with open(folder + "new/" + out_file, 'w') as outfile:
        json.dump(dataset, outfile)

    print(file + " has max length of " + str(max_length))


if __name__ == '__main__':
    prepare_one_dataset("SVHN_data/train/", "digitStruct.mat", "train.json")
    prepare_one_dataset("SVHN_data/test/", "digitStruct.mat", "test.json")
    prepare_one_dataset("SVHN_data/extra/", "digitStruct.mat", "extra.json")