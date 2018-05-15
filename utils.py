import imageio
from skimage.transform import resize
import numpy as np


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    assert i+crop_w < w and j+crop_h < h, "invalid crop_h (and/or crop_w)."
    return resize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])


def specific_crop(x, point, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(point[1])
    i = int(point[0])
    assert j > 0 and i > 0
    assert i+crop_w < w and j+crop_h < h, "invalid crop_h (and/or crop_w) or starting point."
    return resize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])


def transform(image, npx=64, point=None, is_crop=True, resize_w=64):
    if is_crop:
        if point is None:
            cropped_image = center_crop(image, npx, resize_w=resize_w)
        else:
            cropped_image = specific_crop(image, point, npx, resize_w=resize_w)
    else:
        cropped_image = image if npx==resize_w else resize(image, [resize_w, resize_w])
    return np.array(cropped_image)/127.5 - 1.


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return imageio.imread(path, as_gray=True).astype(np.float)
    else:
        return imageio.imread(path).astype(np.float)


def get_image(image_path, image_size, point=None, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, point=point, is_crop=is_crop, resize_w=resize_w)
