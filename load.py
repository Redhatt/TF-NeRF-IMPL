from globals import *


def save_model(model, name_of_file):
    path_to_models = os.path.join(BASE_DIR, "models")
    if os.path.exists(path_to_models):
        path_to_file = os.path.join(path_to_models, name_of_file)
        model.save(path_to_file)
        INFO(f"model saved successfully at {path_to_file}")
    else:
        CRITICAL(f"could not find {path_to_models} directory!!!")


def init_model(input_size=5, output_size=4, layer_size=8, layer_depth=256, level1=10, level2=6):
    input_shape = 3*(1 + 2 * level1) + 2*(1 + 2 * level2)
    input_ = tf.keras.layers.Input(shape=input_shape)

    output = input_
    for i in range(layer_size):
        output = tf.keras.layers.Dense(layer_depth, activation='relu')(output)
        if i % 4 == 0 and i > 0:
            output = tf.concat([output, input_], -1)

    output = tf.keras.layers.Dense(output_size, activation=None)(output)
    model = tf.keras.Model(inputs=input_, outputs=output)

    return model


def load_model(name_of_file=None, level1=10, level2=6):
    if name_of_file is None:
        INFO("Generating new model")
        model = init_model(level1=level1, level2=level2)
        INFO(model.summary())
        return model

    path_to_models = os.path.join(BASE_DIR, "models")
    path_to_file = os.path.join(path_to_models, name_of_file)
    if os.path.exists(path_to_file):
        try:
            model = tf.keras.models.load_model(path_to_file, custom_objects={'Functional': tf.keras.models.Model})
            INFO(f"model loaded successfully from {path_to_file}")
        except OSError:
            ERROR("could not load model, generating new one.", exc_info=True)
            model = init_model(level1=level1, level2=level2)
    else:
        CRITICAL(f"could not find {path_to_file}, generating new one.", exc_info=True)
        model = init_model(level1=level1, level2=level2)

    return model


def load_data(name_of_file='tiny_nerf_data.npz'):
    path_to_data = os.path.join(BASE_DIR, "data")

    if os.path.exists(path_to_data):
        path_to_file = os.path.join(path_to_data, name_of_file)
        data = np.load(path_to_file)
        INFO(f"data loaded successfully from {path_to_file}")
    else:
        CRITICAL(f"could not find {path_to_data} directory!!!")
        data = None

    return data
