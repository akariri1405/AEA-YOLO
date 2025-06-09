import tensorflow as tf
import numpy as np
# Removed deprecated: import tensorflow.contrib.layers as ly
from util_filters import lrelu, rgb2lum, tanh_range, lerp
import cv2
import math

class Filter:
    def __init__(self, net, cfg):
        self.cfg = cfg
        # Specified in child classes:
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name, "Short name must be defined."
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters is not None, "Number of filter parameters not set."
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def extract_parameters(self, features):
        # Returns a tuple of the filter parameters (and mask parameters if needed)
        start = self.get_begin_filter_parameter()
        end = start + self.get_num_filter_parameters()
        return features[:, start:end], features[:, start:end]

    # Should be implemented in child classes
    def filter_param_regressor(self, features):
        assert False, "Child class must implement filter_param_regressor"

    # Process the whole image, without masking
    # Should be implemented in child classes
    def process(self, img, param, defog, IcA):
        assert False, "Child class must implement process"

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    # Apply the whole filter with masking (if enabled)
    def apply(self, img, img_features=None, defog_A=None, IcA=None, specified_parameter=None, high_res=None):
        assert (img_features is None) ^ (specified_parameter is None), \
            "Either img_features or specified_parameter must be provided, but not both."
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            # When no masking is used, the specified parameter is taken directly.
            assert not self.use_masking(), "Masking is enabled but no features provided."
            filter_parameters = specified_parameter
            mask_parameters = tf.zeros(shape=(1, self.get_num_mask_parameters()), dtype=tf.float32)
        if high_res is not None:
            # If processing high-res images separately (not implemented here)
            pass
        debug_info = {}
        if self.debug_info_batched():
            debug_info['filter_parameters'] = filter_parameters
        else:
            debug_info['filter_parameters'] = filter_parameters[0]
        low_res_output = self.process(img, filter_parameters, defog_A, IcA)
        high_res_output = None
        return low_res_output, filter_parameters

    def use_masking(self):
        return self.cfg.masking

    def get_num_mask_parameters(self):
        return 6

    # Compute a mask for spatially varying filter strength.
    def get_mask(self, img, mask_parameters):
        if not self.use_masking():
            print('* Masking Disabled')
            return tf.ones(shape=(1, 1, 1, 1), dtype=tf.float32)
        else:
            print('* Masking Enabled')
        with tf.name_scope('mask'):
            filter_input_range = 5
            assert mask_parameters.shape[1] == self.get_num_mask_parameters()
            # Apply a tanh-range nonlinearity (assumed defined in util_filters)
            mask_parameters = tanh_range(l=-filter_input_range, r=filter_input_range, initial=0)(mask_parameters)
            size = list(map(int, img.shape[1:3]))
            grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)
            shorter_edge = min(size[0], size[1])
            for i in range(size[0]):
                for j in range(size[1]):
                    grid[0, i, j, 0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                    grid[0, i, j, 1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
            grid = tf.constant(grid)
            inp = (grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] +
                   grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] +
                   mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) +
                   mask_parameters[:, None, None, 3, None] * 2)
            inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4, None] / filter_input_range
            mask = tf.sigmoid(inp)
            mask = mask * (mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 + 0.5)
            print('mask', mask.shape)
        return mask

    def visualize_mask(self, debug_info, res):
        # Resize the mask visualization to a given resolution.
        return cv2.resize(debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
                          dsize=res, interpolation=cv2.INTER_NEAREST)

    def draw_high_res_text(self, text, canvas):
        cv2.putText(canvas, text, (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=5)
        return canvas


# ----------------- Filter Subclasses -----------------

class ExposureFilter(Filter):
    def __init__(self, net, cfg):
        super(ExposureFilter, self).__init__(net, cfg)
        self.short_name = 'E'
        self.begin_filter_parameter = cfg.exposure_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(-self.cfg.exposure_range, self.cfg.exposure_range, initial=0)(features)

    def process(self, img, param, defog, IcA):
        # Multiply image by an exponential factor (base 2)
        return img * tf.exp(param[:, None, None, :] * np.log(2))


class UsmFilter(Filter):
    def __init__(self, net, cfg):
        super(UsmFilter, self).__init__(net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param, defog_A, IcA):
        def make_gaussian_2d_kernel(sigma, dtype=tf.float32):
            radius = 12
            x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
            k = tf.exp(-0.5 * tf.square(x / sigma))
            k = k / tf.reduce_sum(k)
            return tf.expand_dims(k, 1) * k

        kernel_i = make_gaussian_2d_kernel(5)
        print('kernel_i.shape', kernel_i.shape)
        kernel_i = tf.tile(kernel_i[:, :, None, None], [1, 1, 1, 1])
        pad_w = (25 - 1) // 2
        padded = tf.pad(img, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
        outputs = []
        for channel_idx in range(3):
            data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
            data_c = tf.nn.conv2d(data_c, kernel_i, [1, 1, 1, 1], 'VALID')
            outputs.append(data_c)
        output = tf.concat(outputs, axis=3)
        img_out = (img - output) * param[:, None, None, :] + img
        return img_out


class UsmFilter_sigma(Filter):
    def __init__(self, net, cfg):
        super(UsmFilter_sigma, self).__init__(net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param, defog_A, IcA):
        def make_gaussian_2d_kernel(sigma, dtype=tf.float32):
            radius = 12
            x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
            k = tf.exp(-0.5 * tf.square(x / sigma))
            k = k / tf.reduce_sum(k)
            return tf.expand_dims(k, 1) * k

        kernel_i = make_gaussian_2d_kernel(param[:, None, None, :])
        print('kernel_i.shape', kernel_i.shape)
        kernel_i = tf.tile(kernel_i[:, :, None, None], [1, 1, 1, 1])
        pad_w = (25 - 1) // 2
        padded = tf.pad(img, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
        outputs = []
        for channel_idx in range(3):
            data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
            data_c = tf.nn.conv2d(data_c, kernel_i, [1, 1, 1, 1], 'VALID')
            outputs.append(data_c)
        output = tf.concat(outputs, axis=3)
        img_out = (img - output) * param[:, None, None, :] + img
        return img_out


class DefogFilter(Filter):
    def __init__(self, net, cfg):
        super(DefogFilter, self).__init__(net, cfg)
        self.short_name = 'DF'
        self.begin_filter_parameter = cfg.defog_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.defog_range)(features)

    def process(self, img, param, defog_A, IcA):
        print('      defog_A:', img.shape)
        print('      IcA:', IcA.shape)
        print('      defog_A:', defog_A.shape)
        tx = 1 - param[:, None, None, :] * IcA
        tx_1 = tf.tile(tx, [1, 1, 1, 3])
        return (img - defog_A[:, None, None, :]) / tf.maximum(tx_1, 0.01) + defog_A[:, None, None, :]


class GammaFilter(Filter):
    def __init__(self, net, cfg):
        super(GammaFilter, self).__init__(net, cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_gamma_range = np.log(self.cfg.gamma_range)
        return tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param, defog_A, IcA):
        param_1 = tf.tile(param, [1, 3])
        return tf.pow(tf.maximum(img, 0.0001), param_1[:, None, None, :])


class ImprovedWhiteBalanceFilter(Filter):
    def __init__(self, net, cfg):
        super(ImprovedWhiteBalanceFilter, self).__init__(net, cfg)
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
        print(mask.shape)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = tf.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        color_scaling *= 1.0 / (1e-5 + 0.27 * color_scaling[:, 0] +
                                0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2])[:, None]
        return color_scaling

    def process(self, img, param, defog, IcA):
        return img * param[:, None, None, :]


class ColorFilter(Filter):
    def __init__(self, net, cfg):
        super(ColorFilter, self).__init__(net, cfg)
        self.curve_steps = cfg.curve_steps
        self.channels = int(net.shape[3])
        self.short_name = 'C'
        self.begin_filter_parameter = cfg.color_begin_param
        self.num_filter_parameters = self.channels * cfg.curve_steps

    def filter_param_regressor(self, features):
        color_curve = tf.reshape(features, shape=(-1, self.channels, self.cfg.curve_steps))[:, None, None, :]
        color_curve = tanh_range(*self.cfg.color_curve_range, initial=1)(color_curve)
        return color_curve

    def process(self, img, param, defog, IcA):
        color_curve = param
        tone_curve_sum = tf.reduce_sum(param, axis=4) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += tf.clip_by_value(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * \
                           color_curve[:, :, :, :, i]
        total_image *= self.cfg.curve_steps / tone_curve_sum
        return total_image


class ToneFilter(Filter):
    def __init__(self, net, cfg):
        super(ToneFilter, self).__init__(net, cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param
        self.num_filter_parameters = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = tf.reshape(features, shape=(-1, 1, self.cfg.curve_steps))[:, None, None, :]
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param, defog, IcA):
        tone_curve = param
        tone_curve_sum = tf.reduce_sum(tone_curve, axis=4) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += tf.clip_by_value(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * \
                           param[:, :, :, :, i]
        total_image *= self.cfg.curve_steps / tone_curve_sum
        return total_image


class VignetFilter(Filter):
    def __init__(self, net, cfg):
        super(VignetFilter, self).__init__(net, cfg)
        self.short_name = 'V'
        self.begin_filter_parameter = cfg.vignet_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tf.sigmoid(features)

    def process(self, img, param):
        return img * 0  # Placeholder implementation

    def get_num_mask_parameters(self):
        return 5

    def get_mask(self, img, mask_parameters):
        with tf.name_scope('mask'):
            filter_input_range = 5
            assert mask_parameters.shape[1] == self.get_num_mask_parameters()
            mask_parameters = tanh_range(l=-filter_input_range, r=filter_input_range, initial=0)(mask_parameters)
            size = list(map(int, img.shape[1:3]))
            grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)
            shorter_edge = min(size[0], size[1])
            for i in range(size[0]):
                for j in range(size[1]):
                    grid[0, i, j, 0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                    grid[0, i, j, 1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
            grid = tf.constant(grid)
            inp = (grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None]) ** 2 + \
                  (grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None]) ** 2 + \
                  mask_parameters[:, None, None, 2, None] - filter_input_range
            inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 3, None] / filter_input_range
            mask = tf.sigmoid(inp)
            mask = mask * (mask_parameters[:, None, None, 4, None] / filter_input_range * 0.5 + 0.5)
            if not self.use_masking():
                print('* Masking Disabled')
                mask = mask * 0 + 1
            else:
                print('* Masking Enabled')
            print('mask', mask.shape)
        return mask

class ContrastFilter(Filter):
    def __init__(self, net, cfg):
        super(ContrastFilter, self).__init__(net, cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tf.tanh(features)

    def process(self, img, param, defog, IcA):
        luminance = tf.minimum(tf.maximum(rgb2lum(img), 0.0), 1.0)
        contrast_lum = -tf.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param[:, :, None, None])


class WNBFilter(Filter):
    def __init__(self, net, cfg):
        super(WNBFilter, self).__init__(net, cfg)
        self.short_name = 'BW'
        self.begin_filter_parameter = cfg.wnb_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tf.sigmoid(features)

    def process(self, img, param, defog, IcA):
        luminance = rgb2lum(img)
        return lerp(img, luminance, param[:, :, None, None])


class LevelFilter(Filter):
    def __init__(self, net, cfg):
        super(LevelFilter, self).__init__(net, cfg)
        self.short_name = 'Le'
        self.begin_filter_parameter = cfg.level_begin_param
        self.num_filter_parameters = 2

    def filter_param_regressor(self, features):
        return tf.sigmoid(features)

    def process(self, img, param):
        lower = param[:, 0]
        upper = param[:, 1] + 1
        lower = lower[:, None, None, None]
        upper = upper[:, None, None, None]
        return tf.clip_by_value((img - lower) / (upper - lower + 1e-6), 0.0, 1.0)


class SaturationPlusFilter(Filter):
    def __init__(self, net, cfg):
        super(SaturationPlusFilter, self).__init__(net, cfg)
        self.short_name = 'S+'
        self.begin_filter_parameter = cfg.saturation_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tf.sigmoid(features)

    def process(self, img, param, defog, IcA):
        img = tf.minimum(img, 1.0)
        hsv = tf.image.rgb_to_hsv(img)
        s = hsv[:, :, :, 1:2]
        v = hsv[:, :, :, 2:3]
        enhanced_s = s + (1 - s) * (0.5 - tf.abs(0.5 - v)) * 0.8
        hsv1 = tf.concat([hsv[:, :, :, 0:1], enhanced_s, hsv[:, :, :, 2:]], axis=3)
        full_color = tf.image.hsv_to_rgb(hsv1)
        param = param[:, :, None, None]
        color_param = param
        img_param = 1.0 - param
        return img * img_param + full_color * color_param
