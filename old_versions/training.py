from globals import *
from load import init_model, load_data, load_model, save_model
from times import Timings

LEVEL = 10


def position_encoding(x, level=10):
    ans = [x]
    for i in range(level):
        for func in (tf.sin, tf.cos):
            ans.append(func((2. ** i) * x))
    return tf.concat(ans, -1)


def get_rays(H, W, focal, c2w):
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_angles(ray_d):
    a = tf.norm(ray_d, axis=-1)[..., None]
    ray_d_norm = ray_d / a
    theta = tf.reduce_sum(ray_d_norm * tf.constant([0., 0., 1.]), -1)
    phi = tf.reduce_sum(ray_d_norm * tf.constant([1., 0., 0.]), -1)
    return theta, phi


def render(model, ray_o, ray_d, near, far, samples=64):
    theta, phi = get_angles(ray_d)
    return render_(model, ray_o, ray_d, theta, phi, near, far, samples)


def render_(model, ray_o, ray_d, theta, phi, near, far, samples):
    def batch_query(input_, chunks=2 ** 15):
        return tf.concat([model(input_[x: x + chunks]) for x in range(0, input_.shape[0], chunks)], 0)

    # generating line
    line = tf.linspace(near, far, samples)
    line += tf.random.uniform(list(ray_o.shape[:-1]) + [samples]) * (far - near) / samples

    # generating points
    points = ray_o[..., None, :] + ray_d[..., None, :] * line[..., :, None]

    # sin and cos generation for high frequency position
    points = position_encoding(points, level=LEVEL)

    # direction concatenation in positions
    theta, phi = theta[..., None], phi[..., None]
    theta = tf.concat((theta,) * samples, -1)
    phi = tf.concat((phi,) * samples, -1)
    points = tf.concat((points, theta[..., None], phi[..., None]), -1)

    points_shape, last_dim = points.shape[:-1], points.shape[-1]
    points = tf.reshape(points, (-1, last_dim))

    # getting output per direction
    output = batch_query(points)
    output = tf.reshape(output, (*points_shape, -1))

    # range scaling
    sigma = tf.nn.relu(output[..., 3])
    color = tf.math.sigmoid(output[..., :3])

    # voluming rendering
    dist = tf.concat((line[..., 1:] - line[..., :-1], tf.broadcast_to([1e10], line[..., :1].shape)), -1)
    alpha = 1.0 - tf.exp(-sigma * dist)
    weights = alpha * tf.math.cumprod(1.0 - alpha + 1e-10, -1, exclusive=True)
    rgb = tf.reduce_sum(weights[..., None] * color, -2)

    return rgb


def train(model, target, ray_o, ray_d, near, far, samples, divr=1, divc=1):
    variables = model.trainable_variables
    theta, phi = get_angles(ray_d)

    # NOTE: 15,000 is float data limit per iteration eg size: (70,70,3)
    R, C = map(int, ray_o.shape[:2])
    stepr = 2 ** int(math.log2(R / divr))
    stepc = 2 ** int(math.log2(C / divc))

    rows = [(r, min(r + stepr, R)) for r in range(0, R, stepr)]
    cols = [(c, min(c + stepc, C)) for c in range(0, C, stepc)]

    for rl, rh in reversed(rows):
        for cl, ch in reversed(cols):
            ray_o_temp = ray_o[rl:rh, cl:ch, ...]
            ray_d_temp = ray_d[rl:rh, cl:ch, ...]
            target_temp = target[rl:rh, cl:ch, ...]
            theta_temp = theta[rl:rh, cl:ch, ...]
            phi_temp = phi[rl:rh, cl:ch, ...]

            with tf.GradientTape() as tape:
                rgb = render_(model, ray_o_temp, ray_d_temp, theta_temp, phi_temp, near, far, samples)
                loss = model.loss(target_temp, rgb)

            gradients = tape.gradient(loss, variables)
            model.optimizer.apply_gradients(zip(gradients, variables))


def main():
    name = "re_write.h5"
    rate = 8e-5
    model = load_model(name, level1=LEVEL, level2=0)
    model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(rate))

    data = load_data()
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    near, far, samples = 2.0, 6.0, 64
    H, W = images[0].shape[:2]

    size = 100
    N = 5000
    train_images, train_poses = images[:size], poses[:size]
    test_images, test_poses = images[size:], poses[size:]
    snrs = []
    iter = []

    t1 = Timings()
    t2 = Timings()
    t3 = Timings()

    E = 'epoch'
    T = 'training'
    R = 'rays'
    D = 'render'

    for i in range(N+1):
        t1.get(E)

        index = np.random.randint(size)
        target = train_images[index]
        pose = train_poses[index]

        t2.get(R)
        ray_o, ray_d = get_rays(H, W, focal, pose)
        t2.get(R)

        t2.get(T)
        train(model, target, ray_o, ray_d, near, far, samples)
        t2.get(T)

        t1.get(E)

        if i % 20 == 0:
            INFO(f"epoch: {i}")
            save_model(model, name)
            test_image = test_images[1]
            test_pose = test_poses[1]

            ro, rd = get_rays(H, W, focal, test_pose)

            t2.get(D)
            rgb = render(model, ro, rd, near, far, samples)
            t2.get(D)

            loss = tf.reduce_mean(model.loss(test_image, rgb))
            snr = -10.0 * tf.math.log(loss) / tf.math.log(10.0)
            snrs.append(snr.numpy())
            iter.append(i)
            INFO(f"iter: {i} snr: {snr}")

            plt.figure(figsize=[10, 4])

            plt.subplot(1, 3, 1)
            img = tf.reshape(rgb, [H, W, -1]).numpy()
            plt.imshow(img)
            plt.title(f"Iteration: {i}")

            plt.subplot(1, 3, 2)
            plt.imshow(test_image)
            plt.title("Actual")

            plt.subplot(1, 3, 3)
            plt.plot(iter, snrs)
            plt.title('SNR')
            plt.savefig("image.png")
            INFO("image saved as image.png")

            t1.info()
            t2.info()


if __name__ == "__main__":
    main()
