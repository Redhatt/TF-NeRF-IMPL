from globals import *
from load import init_model, load_data, load_model, save_model
from times import Timings
from tensorflow.keras import backend as K

LEVEL1 = 10
LEVEL2 = 6


def decay_rate(iteration, total_iter, rate=0.1, high=5e-4, low=5e-5):
    """
    decay function:-> low + (high - low) * exp( -decay * iteration)
    :param iteration: current iteration
    :param total_iter: total iteration
    :param rate: after total_iteration/2 value will be rate% of scaled range
    :param high: high value
    :param low: low value
    :return: calculated learning rate
    """
    decay = (-2.0 / total_iter) * math.log(rate)
    return low + (high - low) * math.exp(-decay * iteration)


def position_encoding(x, level=10):
    """
    encodes sin and cosine values of given vector
    eg: input [x] -> output [x, sin(2^i * x), cos(2^i * x)] i running from 0 to level
    :param x: position tensor
    :param level: frequency level
    :return: tensor
    """
    ans = [x]
    for i in range(level):
        for func in (tf.sin, tf.cos):
            ans.append(func((2. ** i) * x))
    return tf.concat(ans, -1)


def get_rays(H, W, focal, c2w):
    """
    for the given position and focal length returns rays coming out of camera
    :param H: height of image
    :param W: width of image
    :param focal: focal length of camera
    :param c2w: position of camera
    :return: rays origin tensor, rays direction tensor
    """
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_angles(ray_d):
    """
    for given rays direction returns angles from z axis and x axis it makes
    :param ray_d: rays direction tensor
    :return: angles tensor
    """
    a = tf.norm(ray_d, axis=-1)[..., None]
    ray_d_norm = ray_d / a
    theta = tf.reduce_sum(ray_d_norm * tf.constant([0., 0., 1.]), -1)
    phi = tf.reduce_sum(ray_d_norm * tf.constant([1., 0., 0.]), -1)
    return theta, phi


def temp(a):
    b = tf.concat([a[..., :1]] * 4, -1)
    return 2 * tf.sin(1.2 * 50. * b) + 2.2 * tf.cos(1.2 * 101. * b - 2.1) - 3 * tf.cos(231. * b - 7.) - 5 * tf.cos(
        543. * b - 4.)


def inverse_transform_sampling(pdf, near, far, H, W, samples):
    """
    to sample points along a ray we can use this equation:
    y = a + t*b
    where a is origin and b it the direction with parameter t which is scaled distance.
    to sample between near and far we can set t = linspace(near, far, samples)
    but this will produce uniformed sampled points.
    to get a distribution with a desired pdf we can use inverse transform sampling.
    here to reduce high bias towards given pdf few samples of uniform distribution has also been added.
    :param pdf: input pdf distribution
    :param near: near point
    :param far: far point
    :param H: height of image
    :param W: width of image
    :param samples: no of samples on desired pdf
    :return: parametric tensor t having pdf as of the given, no of samples in produced
    """
    line = tf.broadcast_to(tf.linspace(near, far, samples), (H, W, samples))
    given_pdf_samples_size = int(pdf.shape[-1])

    # extracting pdf and cdf
    pdf = pdf / (1e-10 + tf.reduce_sum(pdf, -1)[..., None])
    cdf = tf.math.cumsum(pdf, axis=-1, exclusive=False)

    # interpolating cdf at given given samples (eg: 64 -> 128)
    cdf = tfp.math.batch_interp_regular_nd_grid(
       line[..., None] , line[..., 0, None], line[..., -1, None], cdf, axis=-1
    )

    # inverse transform sampling
    uniform_samples = tf.random.uniform((H, W, samples))
    index = tf.clip_by_value(tf.searchsorted(cdf, uniform_samples, side='right'), 0, samples-1)

    # adding uniform points to not get totally biased
    line_ = tf.linspace(near, far, given_pdf_samples_size)
    line_ += tf.random.uniform((H, W, given_pdf_samples_size)) * (far - near) / given_pdf_samples_size
    line_ = tf.clip_by_value(line_, near, far - 0.01)

    new_line = tf.gather(line, index, axis=2, batch_dims=2)
    new_line = tf.concat([new_line, line_], axis=-1)
    new_line = tf.sort(new_line, axis=-1)
    new_samples = int(new_line.shape[-1])

    # ===================debug graphs=========================
    # temp1 = (new_line - near) / (far - near)  # getting into 0 to 1 range
    # temp1 *= new_samples  # to make sure it gets counted by count bin
    # temp2 = tf.math.bincount(tf.cast(temp1[50, 50], tf.int32), minlength=new_samples)[
    #         :new_samples]  # making bins for discrete pdf
    # bins = tf.math.cumsum(temp2, axis=-1, exclusive=False) / new_samples  # from density to cdf conversion
    #
    # x1 = np.linspace(near, far, given_pdf_samples_size)
    # x2 = np.linspace(near, far, new_samples)
    #
    # # cdf plots
    # plt.subplot(3, 1, 1)
    # plt.plot(line.numpy()[50, 50], cdf.numpy()[50, 50], label="interp. weights cdf")
    # plt.plot(x2, bins.numpy(), label="build weights cdf")
    # plt.title("weights cdf plot")
    # plt.legend()
    # plt.grid()
    #
    # # pdf plot
    # plt.subplot(3, 1, 2)
    # plt.plot(x1, pdf.numpy()[50, 50], label="weights pdf")
    # plt.plot(x2, temp2.numpy() / tf.reduce_sum(temp2).numpy(), label="build weights pdf")
    # plt.title("weights pdf plot")
    # plt.legend()
    # plt.grid()
    #
    # # samples plot
    # plt.subplot(3, 1, 3)
    # plt.scatter(line.numpy()[50, 50], [1] * samples, s=1.8, label="linear sampling")
    # plt.scatter(new_line.numpy()[50, 50], [2] * new_samples, s=1.8, label="In.Tns sampling")
    # plt.title("samples plot")
    # plt.legend()
    # plt.grid()
    #
    # plt.show()

    return new_line, new_samples


def render(model, ray_o, ray_d, near, far, level1=LEVEL1, level2=LEVEL2, samples=64):
    theta, phi = get_angles(ray_d)
    _, w = render_(model, ray_o, ray_d, theta, phi, near, far, samples, level1=level1, level2=level2)
    rgb, _ = render_fine(model, ray_o, ray_d, theta, phi, near, far, 2*samples, pdf=w, level1=level1, level2=level2)
    return rgb


def render_(model, ray_o, ray_d, theta, phi, near, far, samples, level1=LEVEL1, level2=LEVEL2):
    def batch_query(input_, chunks=2 ** 13):
        return tf.concat([model(input_[x: x + chunks]) for x in range(0, input_.shape[0], chunks)], 0)
        # return tf.concat([temp(1.*input_[x: x + chunks]) for x in range(0, input_.shape[0], chunks)], 0)

    # generating line
    H, W = ray_o.shape[:-1]
    line = tf.linspace(near, far, samples)
    line += tf.random.uniform((H, W, samples)) * (far - near) / samples

    # generating points
    points = ray_o[..., None, :] + ray_d[..., None, :] * line[..., :, None]

    # sin and cos generation for high frequency position
    points = position_encoding(points, level=level1)

    # direction concatenation in positions
    theta, phi = theta[..., None], phi[..., None]
    theta = tf.concat((theta,) * samples, -1)[..., None]
    phi = tf.concat((phi,) * samples, -1)[..., None]

    theta = position_encoding(theta, level=level2)
    phi = position_encoding(phi, level=level2)

    points = tf.concat((points, theta, phi), -1)

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

    return rgb, weights


def render_fine(model, ray_o, ray_d, theta, phi, near, far, samples, pdf, level1=LEVEL1, level2=LEVEL2):
    def batch_query(input_, chunks=2 ** 13):
        return tf.concat([model(input_[x: x + chunks]) for x in range(0, input_.shape[0], chunks)], 0)
        # return tf.concat([temp(1.*input_[x: x + chunks]) for x in range(0, input_.shape[0], chunks)], 0)

    # generating line
    H, W = ray_o.shape[:-1]
    line, samples = inverse_transform_sampling(pdf, near, far, H, W, samples)

    # generating points
    points = ray_o[..., None, :] + ray_d[..., None, :] * line[..., :, None]

    # sin and cos generation for high frequency position
    points = position_encoding(points, level=level1)

    # direction concatenation in positions
    theta, phi = theta[..., None], phi[..., None]
    theta = tf.concat((theta,) * samples, -1)[..., None]
    phi = tf.concat((phi,) * samples, -1)[..., None]

    theta = position_encoding(theta, level=level2)
    phi = position_encoding(phi, level=level2)

    points = tf.concat((points, theta, phi), -1)

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

    return rgb, weights


def train(model, target, ray_o, ray_d, near, far, samples, level1=LEVEL1, level2=LEVEL2,lrate=5e-4, divr=1, divc=1):
    variables = model.trainable_variables
    theta, phi = get_angles(ray_d)

    # NOTE: 15,000 is float data limit per iteration eg size: (70,70,3)
    R, C = map(int, ray_o.shape[:2])
    stepr = 2 ** int(math.log2(R / divr))
    stepc = 2 ** int(math.log2(C / divc))
    # stepr = int(R / divr)
    # stepc = int(C / divc)

    rows = [(r, min(r + stepr, R)) for r in range(0, R, stepr)]
    cols = [(c, min(c + stepc, C)) for c in range(0, C, stepc)]
    mean_loss = tf.constant(0.0, dtype=tf.float32)
    count = 0

    for rl, rh in reversed(rows):
        for cl, ch in reversed(cols):
            ray_o_temp = ray_o[rl:rh, cl:ch, ...]
            ray_d_temp = ray_d[rl:rh, cl:ch, ...]
            target_temp = target[rl:rh, cl:ch, ...]
            theta_temp = theta[rl:rh, cl:ch, ...]
            phi_temp = phi[rl:rh, cl:ch, ...]

            with tf.GradientTape() as tape:
                _, w = render_(model, ray_o_temp, ray_d_temp, theta_temp, phi_temp, near, far, samples,
                               level1=level1, level2=level2)
                rgb, _ = render_fine(model, ray_o_temp, ray_d_temp, theta_temp, phi_temp, near, far, 2*samples, w,
                                     level1=level1, level2=level2)
                loss = tf.reduce_mean(model.loss(target_temp, rgb))

            mean_loss += loss
            count += 1

            gradients = tape.gradient(loss, variables)
            K.set_value(model.optimizer.learning_rate, lrate)
            model.optimizer.apply_gradients(zip(gradients, variables))

    return mean_loss / count


def main():
    global LEVEL1, LEVEL2
    name = "car_L6_L5_H_S32.h5"
    LEVEL1 = 3
    LEVEL2 = 3
    lrate = 8e-5

    t1 = Timings()
    t2 = Timings()

    Train = 'Training'
    Render = 'render'
    Plot = 'plotting'
    Stride = 'stride'

    stride = 20
    size = 100
    N = 1000

    model = load_model(name, level1=LEVEL1, level2=LEVEL2)
    model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(lrate))

    data = load_data("car_data.npz")
    images = data['images']
    poses = data['poses']
    focal = tf.constant(data['focal']*1e3, dtype=tf.float32)

    near, far, samples = 18.0, 27.0, 32
    H, W = images[0].shape[:2]

    images = tf.convert_to_tensor(images)

    train_images, train_poses = images[:size], poses[:size]
    test_images, test_poses = images[size:], poses[size:]
    snrs = []
    iters = []
    max_snr = 20
    train_losses = []
    test_losses = []
    data = []

    for i in range(N + 1):
        index = np.random.randint(size)
        target = train_images[index]
        pose = train_poses[index]

        ray_o, ray_d = get_rays(H, W, focal, pose)
        new_rate = decay_rate(i, N, rate=0.3, high=lrate, low=0.4 * lrate)
        # new_rate = lrate

        t2.get(Train)
        train_loss = train(model, target, ray_o, ray_d, near, far, samples,
                           level1=LEVEL1, level2=LEVEL2, lrate=new_rate, divr=5, divc=5)
        t2.get(Train)
        train_losses.append(train_loss)
        data.append((train_loss.numpy(), index))

        iters.append(i)

        if i % stride == 0:
            t2.get(Stride)

            INFO(f"epoch: {i}")
            save_model(model, name)

            test_image = test_images[1]
            test_pose = test_poses[1]

            ro, rd = get_rays(H, W, focal, test_pose)

            t2.get(Render)
            rgb = render(model, ro, rd, near, far, samples=samples, level1=LEVEL1, level2=LEVEL2)
            t2.get(Render)

            loss = tf.reduce_mean(model.loss(test_image, rgb))
            test_losses.append(loss)

            img = rgb

            snr = -10.0 * tf.math.log(loss) / tf.math.log(10.0)
            snrs.append(snr.numpy())
            INFO(f"iter: {i}, snr: {snr}, learning_rate: {new_rate}")
            t2.get(Stride)

            t2.get(Plot)
            plots = 4
            plt.figure(figsize=[3*plots, 3])
            plt.subplot(1, plots, 1)
            plt.imshow(img)
            plt.title(f"Iteration: {i}")

            plt.subplot(1, plots, 2)
            plt.imshow(test_image)
            plt.title("Actual")

            plt.subplot(1, plots, 3)
            plt.plot(iters[::stride], snrs)
            plt.title('SNR')
            plt.grid()

            plt.subplot(1, plots, 4)
            plt.plot(iters, train_losses, label="train loss")
            plt.plot(iters[::stride], test_losses, label="test loss")
            plt.title("train v test loss")
            plt.legend()
            plt.grid()

            plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.85, wspace=0.3, hspace=0.3)

            plt.savefig("image.png")
            plt.close()
            INFO("image saved as image.png")
            t2.get(Plot)

            if snr >= max((*snrs, max_snr)):
                a, b = str(float(snr)).split('.')
                b = b[:min(2, len(b))]
                name_, ext_ = name[:name.rindex(".")], name[name.rindex("."):]
                new_high_name = name_ + "_" + a + "p" + b + ext_;
                save_model(model, new_high_name)

        t1.info()
        t2.info()
        print("="*30 + f" iter:{i+1} <-> loss: {train_loss} <-> lrate: {new_rate} " + "="*30)
        print(data)


if __name__ == "__main__":
    main()
