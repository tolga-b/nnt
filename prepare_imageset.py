"""Python helpers for preprocessing image data for neural networks

# Usage example:
import nnt.prepare_imageset as ppis

# load training image list
fnames, labels = np.loadtxt('.../caffe/data/ilsvrc12/train.txt', str, delimiter=' ').T
labels = map(np.int64, labels)

# set training image path
image_root = 'INPUT_IMAGE_PATH(e.g. .../imagenet/train/)'
output_root = 'OUTPUT_IMAGE_PATH(e.g. .../imagenet/train_min_256/)'
resize_size = 256

ppis.convert_imageset(image_root, fnames, resize_size, output_root)

# In case of an interruption, you can check which files are missing by:
missing = ppis.verify_output(fnames, output_root)
# Then run conversion on missing files to complete it:
ppis.convert_imageset(image_root, missing, resize_size, output_root)

# Note that os.path.join(image_root, fname) for fname in fnames 
# should yield the full path to your input images 
"""
from cv2 import imwrite
from caffe.io import load_image, resize_image
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from timeit import default_timer


def convert_image(im_path, resize_size, input_root, output_root):
    # Resize smaller size of all images to resize_size
    im = load_image(os.path.join(input_root, im_path))
    h, w = map(float, im.shape[:2])
    if not((w <= h and w == resize_size) or (w >= h and h == resize_size)):
        if w < h:
            im = resize_image(im, (int(h / w * resize_size), resize_size), interp_order=1)
        else:
            im = resize_image(im, (resize_size, int(w / h * resize_size)), interp_order=1)
    assert(np.min(im.shape[:2]) == resize_size)
    imwrite(os.path.join(output_root, im_path),
            np.require(255.0 * im[:, :, [2, 1, 0]], dtype=np.uint8))


def convert_imageset(input_root, im_paths, resize_size, output_root):
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # check and create all output dirs
    create_cnt = 0
    for im_path in im_paths:
        head, tail = os.path.split(os.path.join(output_root, im_path))
        if not os.path.isdir(head):
            os.makedirs(head)
            create_cnt += 1
    print('Created {} directories for output.'.format(create_cnt))

    pool = Pool()

    converter = partial(convert_image, resize_size=resize_size,
                        input_root=input_root, output_root=output_root)
    print('Starting conversion...')
    start = default_timer()
    pool.map(converter, im_paths)
    pool.close()
    pool.join()
    elapsed = default_timer() - start
    print('Done in {} secs.'.format(elapsed))


def verify_output(im_paths, output_root):
    missing = []
    for im_path in im_paths:
        if not os.path.isfile(os.path.join(output_root, im_path)):
            missing.append(im_path)
    print('Found {} missing files in output.'.format(len(missing)))
    return missing


# def verify_output_size(im_paths, output_root, resize_size):
#     missing = []
#     for im_path in im_paths:
#         if not os.path.isfile(os.path.join(output_root, im_path)):
#             missing.append(im_path)
#         else:  # the file exists
#             try:
#                 min_shape = np.min(load_image(os.path.join(output_root, im_path)).shape[:2])
#                 if min_shape != resize_size:
#                     missing.append(im_path)
#             except:  # whatever happened should not have happened
#                 missing.append(im_path)

#     print('Found {} missing files in output.'.format(len(missing)))
#     return missing


if __name__ == "__main__":
    pass
