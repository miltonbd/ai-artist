import glob
from PIL import Image
import numpy as np
import  scipy.misc as m
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant','sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

pre_encoded_dir='/media/milton/ssd1/dataset/pascal/VOCdevkit/VOC2012/SegmentationClass/pre_encoded'
pascal_mask='/media/milton/ssd1/dataset/pascal/VOCdevkit/VOC2007/SegmentationClass'

# img=Image.open(glob.glob(pre_encoded_dir+"/**")[10])
# print(np.unique(img))

lbl_path = glob.glob(pascal_mask + "/**")[10]
print(lbl_path)
img1=m.imread(lbl_path)
m.imsave("ori.png",img1)
print(np.unique(img1))
# img1=img1.astype(np.int8)
lbl = encode_segmap(img1)
for index in np.unique(lbl):
    print("{} is {}".format(index,class_names[index]))

# print(np.asarray(img))
# print(img1)

lbl[lbl == 255] = 0
lbl = lbl.astype(float)
lbl = m.imresize(lbl, (375,500), 'nearest', mode='F')
lbl = lbl.astype(int)
for index in np.unique(lbl):
    print("{} is {}".format(index,class_names[index]))

img_resized = decode_segmap(lbl)
print(np.unique(img_resized))
m.imsave("resized.png",img_resized)

