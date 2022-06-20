import matplotlib.pyplot as plt

from globals import *
from load import init_model, load_data, load_model, save_model
from times import Timings
from tensorflow.keras import backend as K

LEVEL1 = 10
LEVEL2 = 6


def decay_rate(iteration, total_iter, rate=0.3, high=5e-4, low=5e-5):
    """
    decay function:-> low + (high - low) * exp( -decay * iteration)
    :param iteration: current iteration
    :param total_iter: total iteration
    :param rate: after total_iteration/2 value will be 10% of scaled range
    :param high: high value
    :param low: low value
    :return: calculated learning rate
    """
    decay = (-2.0 / total_iter) * math.log(rate)
    return low + (high - low) * math.exp(-decay * iteration)


# @tf.function
def position_encoding(x, level=10):
    ans = [0.]*(1 + 2*level)
    ans[0] = x
    count = 1

    for i in range(level):
        for func in (tf.sin, tf.cos):
            ans[count] = func((2. ** i) * x)
            count += 1
    return tf.concat(ans, -1), count


@tf.function
def get_rays(H, W, focal, c2w):
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], (H, W, 3))
    return rays_o, rays_d


@tf.function
def get_angles(ray_d):
    a = tf.norm(ray_d, axis=-1)[..., None]
    ray_d_norm = ray_d / a
    theta = tf.reduce_sum(ray_d_norm * tf.constant([0., 0., 1.]), -1)
    phi = tf.reduce_sum(ray_d_norm * tf.constant([1., 0., 0.]), -1)
    return theta, phi


@tf.function
def render(model, H, W, ray_o, ray_d, near, far, samples=64):
    theta, phi = get_angles(ray_d)
    return render_(model, H, W, ray_o, ray_d, theta, phi, near, far, samples)


@tf.function
def render_(model, H, W, ray_o, ray_d, theta, phi, near, far, samples):
    def batch_query(input_, chunks=2 ** 15):
        return tf.concat([model(input_[x: x + chunks]) for x in tf.range(0, input_.shape[0], chunks)], 0)

    # generating line
    line = tf.linspace(near, far, samples)
    # line += tf.random.uniform(list(ray_o.shape[:-1]) + [samples]) * (far - near) / samples
    # H, W = K.shape(ray_o.shape)[0], K.shape(ray_o.shape)[1]
    # print(H, W)
    line += tf.random.uniform((H, W, samples)) * (far - near) / samples

    # generating points
    points = ray_o[..., None, :] + ray_d[..., None, :] * line[..., :, None]

    # sin and cos generation for high frequency position
    points, l1 = position_encoding(points, level=LEVEL1)

    # direction concatenation in positions
    theta, phi = theta[..., None], phi[..., None]
    theta = tf.concat((theta,) * samples, -1)[..., None]
    phi = tf.concat((phi,) * samples, -1)[..., None]

    theta, l2 = position_encoding(theta, level=LEVEL2)
    phi, l3 = position_encoding(phi, level=LEVEL2)

    points = tf.concat((points, theta, phi), -1)

    last_dim = l1*3 + l2 + l3
    points_shape = (H, W, samples)
    # points_shape, last_dim = points.shape[:-1], points.shape[-1]
    points = tf.reshape(points, (-1, last_dim))

    # getting output per direction
    # output = batch_query(points)
    output = model(points)
    output = tf.reshape(output, (*points_shape, -1))

    # range scaling
    sigma = tf.nn.relu(output[..., 3])
    color = tf.math.sigmoid(output[..., :3])

    # voluming rendering
    dist = tf.concat((line[..., 1:] - line[..., :-1], tf.broadcast_to([1e10], (H, W, 1))), -1)
    alpha = 1.0 - tf.exp(-sigma * dist)
    # weights = alpha * tf.math.cumprod(1.0 - alpha + 1e-10, -1, exclusive=True)
    weights = alpha * tf.exp(-tf.math.cumsum(sigma * dist, axis=-1, exclusive=True))
    rgb = tf.reduce_sum(weights[..., None] * color, -2)

    return rgb


@tf.function
def train(model, R, C, target, ray_o, ray_d, near, far, samples, lrate=5e-4, divr=2, divc=2):
    variables = model.trainable_variables
    theta, phi = get_angles(ray_d)

    # NOTE: 15,000 is float data limit per iteration eg size: (70,70,3)
    stepr = R//divr
    stepc = C//divc

    mean_loss = tf.constant(0.0, dtype=tf.float32)
    count = 0.0

    for rl in tf.range(0, R, stepr):
        for cl in tf.range(0, C, stepc):
            rh = rl + stepr
            ch = cl + stepc

            ray_o_temp = ray_o[rl:rh, cl:ch, ...]
            ray_d_temp = ray_d[rl:rh, cl:ch, ...]
            target_temp = target[rl:rh, cl:ch, ...]
            theta_temp = theta[rl:rh, cl:ch, ...]
            phi_temp = phi[rl:rh, cl:ch, ...]

            H, W = tf.math.minimum(R, rl + stepr) - rl, tf.math.minimum(C, cl + stepc) - cl
            with tf.GradientTape() as tape:
                rgb = render_(model, H, W, ray_o_temp, ray_d_temp, theta_temp, phi_temp, near, far, samples)
                loss = model.loss(target_temp, rgb)

            mean_loss += tf.reduce_mean(loss)
            count += 1.0

            gradients = tape.gradient(loss, variables)
            K.set_value(model.optimizer.learning_rate, lrate)
            model.optimizer.apply_gradients(zip(gradients, variables))

    return mean_loss / count


def main():
    name = "model_L10_L6.h5"
    name = "testing_model.h5"
    lrate = 5e-4
    model = load_model(name, level1=LEVEL1, level2=LEVEL2)
    model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(lrate))

    data = load_data()
    images = data['images']
    poses = data['poses']
    focal = tf.constant(data['focal'], dtype=tf.float32)
    near, far, samples = 2.0, 6.0, 64
    H, W = images[0].shape[:2]

    stride = 10
    size = 100
    N = 500
    train_images, train_poses = images[:size], poses[:size]
    test_images, test_poses = images[size:], poses[size:]
    snrs = []
    iters = []
    train_losses = []
    test_losses = []

    t1 = Timings()
    t2 = Timings()
    t3 = Timings()

    E = 'epoch'
    T = 'training'
    R = 'rays'
    D = 'render'
    F = 'plot'

    for i in range(N+1):
        t1.get(E)

        index = np.random.randint(size)
        target = train_images[index]
        pose = train_poses[index]

        t2.get(R)
        ray_o, ray_d = get_rays(H, W, focal, pose)
        t2.get(R)

        new_rate = decay_rate(i, N, rate=0.3, high=lrate, low=0.08 * lrate)

        t2.get(T)
        train_loss = train(model, H, W, target, ray_o, ray_d, near, far, samples, lrate=new_rate)
        t2.get(T)
        train_losses.append(train_loss)

        t1.get(E)
        iters.append(i)

        if i % stride == 0:
            t1.get(F)
            INFO(f"epoch: {i}")
            save_model(model, name)
            test_image = test_images[1]
            test_pose = test_poses[1]

            ro, rd = get_rays(H, W, focal, test_pose)

            t2.get(D)
            rgb = render(model, H, W, ro, rd, near, far, samples)
            t2.get(D)

            loss = tf.reduce_mean(model.loss(test_image, rgb))
            test_losses.append(loss)

            snr = -10.0 * tf.math.log(loss) / tf.math.log(10.0)
            snrs.append(snr.numpy())
            INFO(f"iter: {i}, snr: {snr}, learning_rate: {new_rate}")

            plt.figure(figsize=[12, 4])

            plt.subplot(1, 4, 1)
            img = tf.reshape(rgb, [H, W, -1]).numpy()
            plt.imshow(img)
            plt.title(f"Iteration: {i}")

            plt.subplot(1, 4, 2)
            plt.imshow(test_image)
            plt.title("Actual")

            plt.subplot(1, 4, 3)
            plt.plot(iters[::stride], snrs)
            plt.title('SNR')
            plt.grid()

            plt.subplot(1, 4, 4)
            plt.plot(iters, train_losses, label="train loss")
            plt.plot(iters[::stride], test_losses, label="test loss")
            plt.title("train v test loss")
            plt.legend()
            plt.grid()

            plt.subplots_adjust(left=0.05, bottom=0.07, right=0.99, top=0.9, wspace=0.3, hspace=0.3)

            plt.savefig("image.png")
            INFO("image saved as image.png")

            t1.get(F)

            t1.info()
            t2.info()


if __name__ == "__main__":
    main()
