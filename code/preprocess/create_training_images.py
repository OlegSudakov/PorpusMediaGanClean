import tifffile
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='path to image')
parser.add_argument('--edgelength', type=int, default=128, help='input batch size')
parser.add_argument('--stride', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--target_name', required=True, help='path to store training images')

opt = parser.parse_args()
print(opt)

img = np.load(str(opt.image))

count = 0

edge_length = opt.edgelength #image dimensions
stride = opt.stride #stride at which images are extracted

N = edge_length
M = edge_length
O = edge_length

I_inc = stride
J_inc = stride
K_inc = stride

images = []
count = 0
for i in range(0, img.shape[0], I_inc):
    for j in range(0, img.shape[1], J_inc):
        for k in range(0, img.shape[2], K_inc):
            subset = img[i:i+N, j:j+N, k:k+O]
            if subset.shape == (N, M, O):
                images.append(subset)
                count += 1
print("Images generated: {}".format(count))
np.save(str(opt.target_name), np.array(images))