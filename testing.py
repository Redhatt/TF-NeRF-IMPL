import matplotlib.pyplot as plt

import globals
from debug import *
from load import *
# from training import position_encoding, get_rays, get_angles, render
import unittest

from trainingV4 import *
from mytiming import Timings


class TestTraining(unittest.TestCase):

    def test_load_model(self):
        model = load_model("re_write25p5_copy.h5")
        model.summary()

    def test_init_model(self):
        model = init_model()
        print(model.summary())

    def test_position_encoding(self):
        x = tf.convert_to_tensor(np.random.rand(100, 100, 3), dtype=tf.float32)
        t = Timings()
        for i in range(100):
            t.get('a')
            val = position_encoding(x)
            t.get('a')
        t.info()
        # debug(val=(val, x, val.shape))

    def test_get_rays(self):
        data = load_data()
        images = data['images']
        poses = data['poses']
        focal = data['focal']
        H, W = images[0].shape[:2]
        pose = poses[0]
        # debug(pose=(pose.shape, pose))
        t = Timings()
        for i in range(100):
            t.get('a')
            ro, rd = get_rays(H, W, focal, pose)
            t.get('a')
        t.info()
        # debug(ro=(ro.shape, ro), rd=(rd.shape, rd))

    def test_get_angles(self):
        ro = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 1., 1.]])
        ro = ro.reshape((2,2,3))
        debug(ro=(ro.shape, ro))
        ro = tf.convert_to_tensor(ro, dtype=tf.float32)
        theta, phi = get_angles(ro)
        debug(theta=(theta.shape, theta), phi=(phi.shape, phi))

    def test_render_(self):
        model = init_model()
        data = load_data()
        images = data['images']
        poses = data['poses']
        focal = tf.constant(data['focal'], dtype=tf.float32)
        near, far, samples = 2.0, 6.0, 5
        H, W = images[0].shape[:2]
        pose = poses[0]
        ro, rd = get_rays(H, W, focal, pose)

        t = Timings()
        for i in range(100):
            t.get('total')
            rgb = render(model, ro, rd, near, far, samples)
            t.get('total')
        t.info()
        # debug(rgb=(rgb.shape, rgb))

    def test_decay_rate(self):
        total_iter = 1000
        rate, high, low = 0.1, 5.e-4, 5e-5
        x = []
        y = []
        for i in range(total_iter):
            x.append(i)
            y.append(decay_rate(i, total_iter, rate, high, low))

        plt.plot(x, y)
        plt.grid()
        plt.axvline()
        plt.axhline()
        plt.show()

    def test_train(self):
        model = init_model()
        model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(0.0005))
        target = np.ones((100, 100, 3), dtype=np.float32)
        ray_o = tf.convert_to_tensor(np.random.rand(100, 100, 3), dtype=tf.float32)
        ray_d = tf.convert_to_tensor(np.random.rand(100, 100, 3), dtype=tf.float32)
        d = Timings()
        R, C = target.shape[0], target.shape[1]
        for i in range(10):
            d.get("total")
            train(model, R, C, target, ray_o, ray_d, near=2., far=6., samples=64)
            d.get("total")
        d.info()

    def test_distribution(self):
        model = init_model("model_L10_L6.h5")
        data = load_data()
        images = data['images']
        poses = data['poses']
        focal = tf.constant(data['focal'], dtype=tf.float32)
        near, far, samples = 2.0, 6.0, 32
        # images = np.array([[[1,1,1], [2,2,2]],
        #                    [[3,3,3], [4,4,4]]],
        #                   dtype=np.float32).reshape((1,2,2,3))
        H, W = images[0].shape[:2]
        pose = poses[0]
        ro, rd = get_rays(H, W, focal, pose)
        rgb = render(model, ro, rd, near, far, samples)

    def test_inverse_transform_sampling(self):
        model = load_model("model_L10_L6_H_S64.h5")
        data = load_data()
        images = data['images']
        poses = data['poses']
        focal = tf.constant(data['focal'], dtype=tf.float32)
        near, far, samples = 2.0, 6.0, 64
        # images = np.arange(50*50*3).reshape((1,50,50,3))
        H, W = images[1].shape[:2]
        pose = poses[1]
        # pose = np.array(
        #     [[-9.3054223e-01,  1.1707554e-01, - 3.4696460e-01, - 1.3986591e+00],
        #      [-3.6618456e-01, - 2.9751042e-01,  8.8170075e-01,  3.5542498e+00],
        #      [7.4505806e-09, 9.4751304e-01, 3.1971723e-01, 1.2888215e+00],
        #      [0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
        #      ], dtype=np.float32
        # )
        print(pose)
        ro, rd = get_rays(H, W, focal, pose)
        theta, phi = get_angles(rd)
        # rgb, w,s = render_(model, ro, rd, theta, phi, near, far, samples=samples, level1=10, level2=6)

        # w = np.zeros((H, W, samples))
        # a = np.linspace(0., 1., samples)
        # b = np.sin(2*np.pi*2*(a - 0.125))
        # b[b<0] = 0
        # w = tf.broadcast_to(b, w.shape)
        # w = tf.cast(w, dtype=tf.float32)

        a = render(model, ro, rd, near, far, samples=samples, level1=10, level2=6)
        # plt.imshow(a.numpy())
        # plt.show()
        # d = Timings()
        # d.get("total")
        # line = inverse_transform_sampling(w, s, near, far, H, W, samples)
        # d.get("total")
        # d.info()

# def test_render_():
#     data = load_data("tiny_nerf_data.npz")
#     H, W = data['images'].shape[:2]
#     focal = data['focal']
#     poses = data['poses']
#     model = init_model()
#
#     focal = tf.constant(1 / focal, dtype=tf.float32)
#     pose = poses[0]
#     H, W = 2, 2
#     ro, rd = get_rays(H, W, focal, pose)
#     render(model, ro, rd, 2., 6., 64)
#
#     # model = load_model()
#     # ray_o = tf.convert_to_tensor(
#     #     []
#     # )
#
#     # model, ray_o, ray_d, theta, phi, near, far, samples
#
#
# if __name__ == "__main__":
#     test_render_()
