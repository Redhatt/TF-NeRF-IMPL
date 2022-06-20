import os

import numpy as np
from PIL import Image
from globals import *
from load import *

path = "C:\\Users\\Danish Amin\\Downloads\\3d models\\images"
save = "data\\car_data.npz"


def load_image(file):
    img = Image.open(file)
    img.load()
    data = np.asarray(img, dtype="float32") / 255
    return data


def decode(name):
    value = name[:name.rindex(".")]
    values = list(map(float, value.split("_")))
    return np.array(values, dtype=np.float32)


def cartToc2w(pose):
    trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    rot_phi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    pos = pose[1:4]
    x, y, z = pos[0], pos[1], pos[2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y/(x + 1e-10))
    phi = np.arctan(np.sqrt(x**2 + y**2)/z)

    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def main():
    poses, images, focal = [], [], None
    for name in os.listdir(path):
        file = os.path.join(path, name)
        if not os.path.isfile(file):
            continue

        pose, image = decode(name), load_image(file)
        # poses.append(pose[..., None])
        poses.append(cartToc2w(pose))
        images.append(image)

        if focal is None:
            focal = pose[0].astype(np.float32)

    poses = np.stack(poses).astype(np.float32)
    images = np.stack(images).astype(np.float32)

    np.savez(save, images=images, poses=poses, focal=focal)


main()
