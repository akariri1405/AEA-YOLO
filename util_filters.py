import math
import cv2
import tensorflow as tf
import os
import sys
import numpy as np
from skimage import color 

# Define state dimensions
STATE_REWARD_DIM = 0
STATE_STOPPED_DIM = 1
STATE_STEP_DIM = 2
STATE_DROPOUT_BEGIN = 3

def get_expert_file_path(expert):
    # Build path using os.path.join for Windows compatibility
    expert_path = os.path.join('data', 'artists', f'fk_{expert}')
    return expert_path

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keepdims=True)
    return (x - mean) / tf.sqrt(var + epsilon)

def enrich_image_input(cfg, net, states):
    if cfg.img_include_states:
        print("Enriching with states; shape:", states.shape)
        states = states[:, None, None, :] + (net[:, :, :, 0:1] * 0)
        net = tf.concat([net, states], axis=3)
    return net

# A simple dictionary subclass that allows attribute access.
class Dict(dict):
    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, key, value):
        self.__setitem__(key, value)
    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})
    def __delattr__(self, item):
        self.__delitem__(item)
    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]

def make_image_grid(images, per_row=8, padding=2):
    npad = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant', constant_values=1.0)
    assert images.shape[0] % per_row == 0, "The total number of images must be a multiple of per_row"
    num_rows = images.shape[0] // per_row
    image_rows = []
    for i in range(num_rows):
        image_rows.append(np.hstack(images[i * per_row:(i + 1) * per_row]))
    return np.vstack(image_rows)

def get_image_center(image):
    if image.shape[0] > image.shape[1]:
        start = (image.shape[0] - image.shape[1]) // 2
        image = image[start:start + image.shape[1], :]
    if image.shape[1] > image.shape[0]:
        start = (image.shape[1] - image.shape[0]) // 2
        image = image[:, start:start + image.shape[0]]
    return image

def rotate_image(image, angle):
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    # Extend to 3x3 matrix
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    image_w2, image_h2 = image_size[0] * 0.5, image_size[1] * 0.5
    coords = np.array([
        [-image_w2, image_h2],
        [image_w2, image_h2],
        [-image_w2, -image_h2],
        [image_w2, -image_h2]
    ])
    rot_coords = np.dot(coords, rot_mat[:2, :2].T)
    x_coords = rot_coords[:, 0]
    y_coords = rot_coords[:, 1]
    right_bound = max(x_coords)
    left_bound = min(x_coords)
    top_bound = max(y_coords)
    bot_bound = min(y_coords)
    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    trans_mat = np.array([[1, 0, int(new_w * 0.5 - image_w2)],
                          [0, 1, int(new_h * 0.5 - image_h2)],
                          [0, 0, 1]])
    affine_mat = np.dot(trans_mat, rot_mat)[:2, :]
    result = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return result

def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = h if (w < h) else w
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (bb_w - 2 * x, bb_h - 2 * y)

def crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
    if width > image_size[0]:
        width = image_size[0]
    if height > image_size[1]:
        height = image_size[1]
    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    return image[y1:y2, x1:x2]

def rotate_and_crop(image, angle):
    image_width, image_height = image.shape[1], image.shape[0]
    image_rotated = rotate_image(image, angle)
    new_dims = largest_rotated_rect(image_width, image_height, math.radians(angle))
    image_rotated_cropped = crop_around_center(image_rotated, *new_dims)
    return image_rotated_cropped

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

def double_lrelu(x, leak=0.1, name="double_lrelu"):
    with tf.name_scope(name):
        return tf.minimum(tf.maximum(leak * x, x), leak * x - (leak - 1))

def leaky_clamp(x, lower, upper, leak=0.1, name="leaky_clamp"):
    with tf.name_scope(name):
        x_norm = (x - lower) / (upper - lower)
        return tf.minimum(tf.maximum(leak * x_norm, x_norm), leak * x_norm - (leak - 1)) * (upper - lower) + lower

class Tee(object):
    def __init__(self, name):
        self.file = open(name, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    def __del__(self):
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()
    def write_to_file(self, data):
        self.file.write(data)
    def flush(self):
        self.file.flush()

def rgb2lum(image):
    lum = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
    return lum[:, :, :, None]

def tanh01(x):
    return tf.tanh(x) * 0.5 + 0.5

def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):
        def activation(x):
            if initial is not None:
                bias = math.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left
        return activation
    return get_activation(l, r, initial)

def merge_dict(a, b):
    ret = a.copy()
    for key, val in b.items():
        if key in ret:
            assert False, 'Item ' + key + ' already exists'
        else:
            ret[key] = val
    return ret

def lerp(a, b, l):
    return (1 - l) * a + l * b

def read_tiff16(fn):
    import tifffile
    img = tifffile.imread(fn)
    if img.dtype == np.uint8:
        depth = 8
    elif img.dtype == np.uint16:
        depth = 16
    else:
        print("Warning: unsupported data type {}. Assuming 16-bit.".format(img.dtype))
        depth = 16
    return (img * (1.0 / (2**depth - 1))).astype(np.float32)

def load_config(config_name):
    scope = {}
    exec('from config_{} import cfg'.format(config_name), scope)
    return scope['cfg']

def get_artist_batch(folder, size=128, num=64):
    js = os.listdir(folder)
    np.random.shuffle(js)
    imgs = np.zeros((num, size, size, 3))
    for i, jpg in enumerate(js[:num]):
        img = cv2.imread(os.path.join(folder, jpg))
        img = get_image_center(img) / 255.
        imgs[i] = cv2.resize(img, dsize=(size, size))
    return imgs

def show_artist_subnails(folder, size=128, num_row=8, num_column=8):
    imgs = get_artist_batch(folder, size, num_row * num_column)
    return make_image_grid(imgs, per_row=num_row)

def np_tanh_range(l, r):
    def get_activation(left, right):
        def activation(x):
            return np.tanh(x) * (right - left) + left
        return activation
    return get_activation(l, r)

class WB2:
    def filter_param_regressor(self, features):
        log_wb_range = np.log(5)
        color_scaling = np.exp(np_tanh_range(-log_wb_range, log_wb_range)(features[:, :3]))
        return color_scaling
    def process(self, img, param):
        lum = (img[:, :, :, 0] * 0.27 + img[:, :, :, 1] * 0.67 +
               img[:, :, :, 2] * 0.06 + 1e-5)[:, :, :, None]
        tmp = img * param[:, None, None, :]
        tmp = tmp / (tmp[:, :, :, 0] * 0.27 + tmp[:, :, :, 1] * 0.67 +
                     tmp[:, :, :, 2] * 0.06 + 1e-5)[:, :, :, None] * lum
        return tmp

def degrade_images_in_folder(folder, dst_folder_suffix, LIGHTDOWN=True, UNBALANCECOLOR=True):
    js = os.listdir(folder)
    dst_folder = folder + '-' + dst_folder_suffix
    try:
        os.mkdir(dst_folder)
    except:
        print('Directory exists!')
    print('Saving degraded images in ' + dst_folder)
    num = 3
    for j in js:
        img = cv2.imread(os.path.join(folder, j)) / 255.
        if LIGHTDOWN:
            for _ in range(num - 1):
                out = np.power(img, np.random.uniform(0.4, 0.6)) * np.random.uniform(0.25, 0.5)
                cv2.imwrite(os.path.join(dst_folder, f'L{_}-' + j), out * 255.)
            out = img * img
            out = out * (1.0 / out.max())
            cv2.imwrite(os.path.join(dst_folder, f'L{num}-' + j), out * 255.)
        if UNBALANCECOLOR:
            filter = WB2()
            outs = np.array([img] * num)
            features = np.abs(np.random.rand(num, 3))
            for _, out in enumerate(filter.process(outs, filter.filter_param_regressor(features))):
                out /= out.max()
                out *= np.random.uniform(0.7, 1)
                cv2.imwrite(os.path.join(dst_folder, f'C{_}-' + j), out * 255.)

def vis_images_and_indexs(images, features, dir, name):
    id_imgs = []
    for feature in features:
        img = np.ones((64, 64, 3))
        cv2.putText(img, str(feature), (4, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (1.0, 0.0, 0.0))
        id_imgs.append(img)
    id_imgs = np.stack(id_imgs, axis=0)
    vis_imgs = np.vstack([images, id_imgs])
    image = make_image_grid(vis_imgs, per_row=images.shape[0])
    vis_dir = dir
    try:
        os.mkdir(vis_dir)
    except:
        pass
    cv2.imwrite(os.path.join(vis_dir, name + '.png'), image[:, :, ::-1] * 255.0)

def read_set(name):
    if name == 'u_test':
        fn = os.path.join('data', 'folds', 'FiveK_test.txt')
        need_reverse = False
    elif name == 'u_amt':
        fn = os.path.join('data', 'folds', 'FiveK_test_AMT.txt')
        need_reverse = False
    elif name == '5k':
        return list(range(1, 5001))
    elif name == '2k_train':
        fn = os.path.join('data', 'folds', 'FiveK_train_first2k.txt')
        need_reverse = False
    elif name == '2k_target':
        fn = os.path.join('data', 'folds', 'FiveK_train_second2k.txt')
        need_reverse = False
    else:
        assert False, name + ' not found'
    l = []
    with open(fn, 'r') as f:
        for i in f:
            if i[0] != '#':
                try:
                    i = int(i)
                    l.append(i)
                except Exception as e:
                    print(e)
                    pass
    if need_reverse:
        l = list(set(range(1, 5001)) - set(l))
    return l
