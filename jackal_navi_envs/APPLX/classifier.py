import os
import csv

import time
import yaml
import shutil
import pickle
import argparse
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

SCAN_RANGE = 1.5 * np.pi
SCAN_NUM = 720


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--training_params_filename',
                        type=str,
                        default='train_scan_classifier.yaml',
                        help='Name of filename defining the learning params')

    args = parser.parse_args()

    config = yaml.load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    args.lr = float(args.lr)
    if args.centering_on_gp:
        args.cropping = False

    args.rslts_dir = os.path.join("..", "rslts", "{}".format(time.strftime("%Y-%m-%d-%H-%M-%S")))
    os.makedirs(args.rslts_dir, exist_ok=True)
    shutil.copyfile(args.training_params_filename, os.path.join(args.rslts_dir, args.training_params_filename))

    args.Dy = len(args.used_context)

    return args


def plot_scan(scan, gp, save_path, args):
    plt.polar(np.linspace(-SCAN_RANGE / 2, SCAN_RANGE / 2, len(scan)), scan)
    gp = np.concatenate([np.zeros((1, 2)), gp], axis=0)
    plt.polar(np.arctan2(gp[:, 1], gp[:, 0]), np.linalg.norm(gp, axis=-1))
    plt.gca().set_theta_zero_location("N")
    plt.gca().set_ylim([0, args.clipping])
    plt.savefig(save_path)
    plt.close("all")


def preprocess_scan(scan, args):
    i = 0
    while i < len(scan):
        if scan[i] != np.inf:
            i += 1
            continue

        j = i + 1
        while j < len(scan) and scan[j] == np.inf:
            j += 1

        if i == 0:
            scan[i:j] = 0.05 * np.ones(j - i)
        elif j == len(scan):
            scan[i:j] = 0.05 * np.ones(j - i)
        else:
            scan[i:j] = np.linspace(scan[i - 1], scan[j], j - i + 1)[1:]
        i = j

    scan = scan.clip(0, args.clipping)
    return scan


def preprocess_gp(gp):
    gp_clip = []
    node_idx = 1
    cum_dist = 0
    for pt, pt_next in zip(gp.T[:-1], gp.T[1:]):
        cum_dist += np.linalg.norm(pt - pt_next)
        if node_idx * 0.2 < cum_dist:
            gp_clip.append(pt_next)
            node_idx += 1
            if node_idx == 11:
                break
    while node_idx < 11:
        node_idx += 1
        gp_clip.append(gp.T[-1])
    gp_clip = np.array(gp_clip)
    return gp_clip


def get_dataset(args, draw_data=False):
    scan_train, gp_train, y_train = [], [], []
    scan_test, gp_test, y_test = [], [], []

    for y, fname in enumerate(args.used_context):
        fname_ = os.path.join("../bag_files", fname + ".pkl")
        if not os.path.exists(fname_):
            continue

        scans = []
        gps = []
        ys = []
        with open(fname_, "rb") as f:
            data = pickle.load(f, encoding='latin1')
        for scan, gp in data:
            scan = preprocess_scan(scan, args)
            gp = preprocess_gp(gp)
            scans.append(scan)
            gps.append(gp)
            ys.append(y)

        if draw_data:
            fig_dir = os.path.join("..", "data_plots", fname)
            os.makedirs(fig_dir, exist_ok=True)
            for i, (scan, gp) in enumerate(zip(scans, gps)):
                plot_scan(scan, gp, os.path.join(fig_dir, str(i)), args)

        # # use first and last 10 % as testing data
        num_X = len(scans)
        if not args.full_train:
            # scan_test.extend(np.concatenate([scans[:num_X // 10], scans[-num_X // 10:]]))
            # gp_test.extend(np.concatenate([gps[:num_X // 10], gps[-num_X // 10:]]))
            # y_test.extend(ys[:num_X // 10] + ys[-num_X // 10:])
            # scan_train.extend(scans[num_X // 10:-num_X // 10])
            # gp_train.extend(gps[num_X // 10:-num_X // 10])
            # y_train.extend(ys[num_X // 10:-num_X // 10])
            scans, gps = shuffle(scans, gps)
            scan_test.extend(scans[:num_X // 5])
            gp_test.extend(gps[:num_X // 5])
            y_test.extend(ys[:num_X // 5])
            scan_train.extend(scans[num_X // 5:])
            gp_train.extend(gps[num_X // 5:])
            y_train.extend(ys[num_X // 5:])
        else:
            scan_train.extend(scans)
            gp_train.extend(gps)
            y_train.extend(ys)

    # print stats
    scans = np.array(scan_train + scan_test)
    y = np.array(y_train + y_test)

    scan_train = np.array(scan_train)
    gp_train = np.array(gp_train)
    y_train = np.array(y_train)
    scan_test = np.array(scan_test)
    gp_test = np.array(gp_test)
    y_test = np.array(y_test)

    print("Num of train", len(scan_train))
    print("Num of test", len(scan_test))

    if draw_data:
        plt.figure()
        plt.hist(scans.reshape((-1), 1), bins="auto")
        plt.title("Laser scan distribution")
        plt.ylim([0, 10000])
        plt.savefig("../data_plots/scan_distribution")

        plt.figure()
        (unique, counts) = np.unique(y, return_counts=True)
        plt.bar(unique, counts)
        plt.title("label_distribution")
        plt.savefig("../data_plots/label_distribution")

    return [scan_train, gp_train], y_train, [scan_test, gp_test], y_test


class ScanClassifier(object):
    def __init__(self, args):
        self.Dx = args.Dx
        self.Dy = args.Dy

        self.used_context = args.used_context

        self.use_EDL = args.use_EDL
        self.dropout_rate = args.dropout_rate
        self.scan_Dhs = args.scan_Dhs
        self.gp_Dhs = args.gp_Dhs
        self.use_conv1d = args.use_conv1d
        self.kernel_sizes = args.kernel_sizes
        self.filter_sizes = args.filter_sizes
        self.strides = args.strides

        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.full_train = args.full_train
        self.rslts_dir = args.rslts_dir

        self.use_weigth = args.use_weigth
        self.clipping = args.clipping
        self.centering_on_gp = args.centering_on_gp
        self.cropping = args.cropping
        self.theta_noise_scale = args.theta_noise_scale
        self.noise = args.noise
        self.noise_scale = args.noise_scale
        self.flipping = args.flipping
        self.translation = args.translation
        self.translation_scale = args.translation_scale
        self.mode_inited = False

    def _init_model(self):
        tf.keras.backend.set_learning_phase(1)
        self.scan_ph = tf.placeholder(tf.float32, shape=(None, self.Dx))
        self.gp_ph = tf.placeholder(tf.float32, shape=(None, 10, 2))
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.annealing_step_ph = tf.placeholder(dtype=tf.int32)
        self.dropout_rate_ph = tf.placeholder(dtype=tf.float32)
        self.weight_ph = tf.placeholder(tf.float32, shape=(None,))

        self.scan_encoder, self.gp_encoder, self.classify_layers = [], [], []
        if self.use_conv1d:
            for kernel_size, filter_size, stride in zip(self.kernel_sizes, self.filter_sizes, self.strides):
                self.scan_encoder.append(tf.keras.layers.Conv1D(filter_size, kernel_size, strides=stride,
                                                                 activation="relu"))
            self.scan_encoder.append(tf.keras.layers.Flatten())
        else:
            for Dh in self.scan_Dhs:
                self.scan_encoder.append(tf.keras.layers.Dense(Dh, activation="relu"))

        self.gp_encoder.append(tf.keras.layers.Flatten())
        for Dh in self.gp_Dhs:
            self.gp_encoder.append(tf.keras.layers.Dense(Dh, activation="relu"))

        self.classify_layers.append(tf.keras.layers.Dropout(rate=self.dropout_rate_ph))
        self.classify_layers.append(tf.keras.layers.Dense(self.Dy))

        scan_h = self.scan_ph
        if self.use_conv1d:
            scan_h = scan_h[..., tf.newaxis]
        for layer in self.scan_encoder:
            scan_h = layer(scan_h)

        if self.gp_Dhs == [0]:
            gp_h = tf.zeros_like(scan_h[:, :0])
        else:
            gp_h = self.gp_ph
            for layer in self.gp_encoder:
                gp_h = layer(gp_h)
            
        h = tf.concat([scan_h, gp_h], axis=-1)
        for layer in self.classify_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                h = layer(h, training=False)
            else:
                h = layer(h)

        global_step_ = tf.Variable(initial_value=0, name='global_step', trainable=False)
        if self.use_EDL:
            self.evidence = tf.nn.softplus(h)
            self.alpha = self.evidence + 1
            self.uncertainty = self.Dy / tf.reduce_sum(self.alpha, axis=-1)
            self.confidence = 1 - self.uncertainty
            self.prob = self.alpha / tf.reduce_sum(self.alpha, axis=-1, keepdims=True)
            self.pred = tf.argmax(self.alpha, axis=-1, output_type=tf.int32)

            def KL(alpha, K):
                beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
                S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)

                KL = tf.reduce_sum((alpha - beta) * (tf.digamma(alpha) - tf.digamma(S_alpha)), axis=1, keepdims=True) +\
                    tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keepdims=True) + \
                    tf.reduce_sum(tf.lgamma(beta), axis=1, keepdims=True) - \
                    tf.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
                return KL

            def expected_cross_entropy(p, alpha, K, global_step, annealing_step):
                if self.use_weigth:
                    p = p * self.weight_ph[:, tf.newaxis]
                loglikelihood = tf.reduce_mean(
                    tf.reduce_sum(p * (tf.digamma(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.digamma(alpha)), 1,
                                  keepdims=True))
                KL_reg = tf.minimum(1.0, tf.cast(global_step / annealing_step, tf.float32)) * KL(
                    (alpha - 1) * (1 - p) + 1, K)
                return loglikelihood + KL_reg

            label = tf.one_hot(self.label_ph, self.Dy)
            loss = expected_cross_entropy(label, self.alpha, self.Dy, global_step_, self.annealing_step_ph)
            self.loss = tf.reduce_mean(loss)

        else:
            logits = h

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph, logits=logits)
            if self.use_weigth:
                loss = loss * self.weight_ph
            self.loss = tf.reduce_mean(loss)
            self.pred = tf.argmax(logits, axis=-1, output_type=tf.int32)

        pred_correctness = tf.equal(self.pred, self.label_ph)
        self.acc = tf.reduce_mean(tf.cast(pred_correctness, tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step_)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.keras.backend.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        # self.writer = tf.summary.FileWriter(self.rslts_dir)

    def _save_model(self, epoch_num):
        model_dir = os.path.join(self.rslts_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        params = {}
        for i, module in enumerate([self.scan_encoder, self.gp_encoder, self.classify_layers]):
            for j, layer in enumerate(module):
                weights = layer.get_weights()
                params["weights_{}_{}".format(i, j)] = weights
        with open(os.path.join(model_dir, "model_{}.pickle".format(epoch_num)), "wb") as f:
            pickle.dump(params, f, protocol=2)

    def _load_model(self, fname):
        if not self.mode_inited:
            self._init_model()
            self.mode_inited = True
        with open(fname, "rb") as f:
            params = pickle.load(f)
        for i, module in enumerate([self.scan_encoder, self.gp_encoder, self.classify_layers]):
            if module == self.gp_encoder and self.gp_Dhs == [0]:
                continue
            for j, layer in enumerate(module):
                weights = params["weights_{}_{}".format(i, j)]
                layer.set_weights(weights)

    def _data_augment(self, scans, gps, training=False):
        # scans.shape = (batch_size, D_scan)
        # gps.shape = (batch_size, 10, 2)
        scan_ori, gp_ori = scans.copy(), gps.copy()
        scans = scans.copy()
        scans = np.clip(scans, 0, self.clipping)
        scans /= self.clipping
        data_valid = np.full(len(scans), True)

        if training:
            batch_size = scans.shape[0]
            if self.flipping:
                is_flipped = np.random.rand(batch_size) < 0.5
                scans[is_flipped] = np.flip(scans[is_flipped], axis=-1)
                gps[is_flipped, :, 1] = -gps[is_flipped, :, 1]
            if self.centering_on_gp:
                theta = np.arctan2(gps[:, :, 1], gps[:, :, 0])
                avg_theta = np.mean(theta[:, :5], axis=-1)
                start_idxes = (avg_theta / SCAN_RANGE * SCAN_NUM).astype(int) + SCAN_NUM // 2 - self.Dx // 2
                data_valid = np.logical_and(start_idxes >= 0, start_idxes + self.Dx < SCAN_NUM)
                start_idxes[np.logical_not(data_valid)] = 0
                scans = np.array([scans[i, idx:idx + self.Dx] for i, idx in enumerate(start_idxes)])
            if self.cropping:
                theta_noise = np.random.uniform(-self.theta_noise_scale, self.theta_noise_scale, size=batch_size)
                theta_noise = theta_noise / 180 * np.pi
                start_idxes = (theta_noise / SCAN_RANGE * SCAN_NUM).astype(int) + (SCAN_NUM - self.Dx) // 2
                theta = np.arctan2(gps[:, :, 1], gps[:, :, 0])
                r = np.linalg.norm(gps, axis=-1)
                theta = theta - theta_noise[:, np.newaxis]
                gps = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)
                scans = np.array([scans[i, idx:idx + self.Dx] for i, idx in enumerate(start_idxes)])
            if self.noise:
                scale = self.noise_scale / self.clipping
                scan_noise = np.random.uniform(-scale, scale, size=scans.shape)
                gp_noise = np.random.uniform(-scale, scale, size=gps.shape)
                scans, gps = scans + scan_noise, gps + gp_noise
        else:
            if self.centering_on_gp:
                thetas = np.arctan2(gps[:, :, 1], gps[:, :, 0])[:, :5]
                avg_thetas = np.zeros(thetas.shape[0])
                for i, theta in enumerate(thetas):
                    if np.any(theta < -SCAN_RANGE / 2) and np.any(theta > SCAN_RANGE / 2):
                        theta[theta < 0] += 2 * np.pi
                        avg_theta = np.mean(theta)
                        if avg_theta > np.pi:
                            avg_theta -= 2 * np.pi
                    else:
                        avg_theta = np.mean(theta)
                    avg_thetas[i] = avg_theta
                start_idxes = (avg_thetas / SCAN_RANGE * SCAN_NUM).astype(int) + SCAN_NUM // 2 - self.Dx // 2
                new_scans = np.zeros((scans.shape[0], self.Dx))
                for i, (idx, scan) in enumerate(zip(start_idxes, scans)):
                    if idx < 0 or idx + self.Dx >= len(scan):
                        data_valid[i] = False
                        new_scans[i] = np.zeros(self.Dx)
                    else:
                        new_scans[i] = scan[idx:idx + self.Dx]
                scans = new_scans
            if self.cropping:
                i = (SCAN_NUM - self.Dx) // 2
                scans = scans[:, i:i + self.Dx]

        if not self.centering_on_gp:
            thetas = np.arctan2(gps[:, :, 1], gps[:, :, 0])[:, :5]
            avg_thetas = np.zeros(thetas.shape[0])
            for i, theta in enumerate(thetas):
                if np.any(theta < -SCAN_RANGE / 2) and np.any(theta > SCAN_RANGE / 2):
                    theta[theta < 0] += 2 * np.pi
                    avg_theta = np.mean(theta)
                    if avg_theta > np.pi:
                        avg_theta -= 2 * np.pi
                else:
                    avg_theta = np.mean(theta)
                avg_thetas[i] = avg_theta
            data_valid = np.abs(avg_thetas) < np.pi / 3

        # for x1, x2, x3, x4 in zip(scans, gps, scan_ori, gp_ori):
        #     plt.polar(np.linspace(-SCAN_RANGE / 2, SCAN_RANGE / 2, len(x3)), x3)
        #     x4 = np.concatenate([np.zeros((1, 2)), x4], axis=0)
        #     plt.polar(np.arctan2(x4[:, 1], x4[:, 0]), np.linalg.norm(x4, axis=-1))
        #
        #     x1 *= self.clipping
        #     plt.polar(np.linspace(-SCAN_RANGE / 2, SCAN_RANGE / 2, len(x1)) * self.Dx / SCAN_NUM, x1)
        #     x2 = np.concatenate([np.zeros((1, 2)), x2], axis=0)
        #     plt.polar(np.arctan2(x2[:, 1], x2[:, 0]), np.linalg.norm(x2, axis=-1))
        #
        #     plt.gca().set_theta_zero_location("N")
        #     plt.gca().set_ylim([0, self.clipping])
        #     plt.show()
        return scans, gps, data_valid

    def _translation(self, scans, gps):
        def find_intersect(x1, y1, x2, y2, angle):
            A1, B1, C1 = y1 - y2, x2 - x1, x2 * y1 - x1 * y2
            A2, B2, C2 = np.tan(angle), -1, 0
            x = (B2 * C1 - B1 * C2) / (A1 * B2 - A2 * B1)
            y = (A1 * C2 - A2 * C1) / (A1 * B2 - A2 * B1)
            return x, y

        new_scans, new_gps = [], []
        for scan, gp in zip(scans, gps):
            trans_x, trans_y = np.random.uniform(low=-self.translation_scale, high=self.translation_scale, size=2)
            # new scan
            angles = np.linspace(-SCAN_RANGE / 2, SCAN_RANGE / 2, len(scan))
            x = np.cos(angles) * scan
            y = np.sin(angles) * scan
            x -= trans_x
            y -= trans_y
            new_angles = np.arctan2(y, x)
            new_scan = []
            for angle_i in angles:
                scan_len = []
                for j in range(len(new_angles) - 1):
                    new_angle_j = new_angles[j]
                    new_angle_jp1 = new_angles[j + 1]
                    if (new_angle_j - angle_i) * (new_angle_jp1 - angle_i) > 0:
                        # no intersection
                        continue
                    x_j, y_j = x[j], y[j]
                    x_jp1, y_jp1 = x[j + 1], y[j + 1]
                    if (new_angle_j - angle_i) * (new_angle_jp1 - angle_i) < 0:
                        # exists intersection, find out where it is
                        if np.sqrt((x_j - x_jp1) ** 2 + (y_j - y_jp1) ** 2) < 0.5:
                            # two points are close, they are on the same object
                            inter_x, inter_y = find_intersect(x_j, y_j, x_jp1, y_jp1, angle_i)
                        else:
                            # two are far away from each other, on the different object
                            if (x_j ** 2 + y_j ** 2) > (x_jp1 ** 2 + y_jp1 ** 2):
                                if j > 0:
                                    inter_x, inter_y = find_intersect(x_j, y_j, x[j - 1], y[j - 1], angle_i)
                                else:
                                    inter_x, inter_y = x_j, y_j
                            else:
                                if j < len(new_angles) - 2:
                                    inter_x, inter_y = find_intersect(x_jp1, y_jp1, x[j + 2], y[j + 2], angle_i)
                                else:
                                    inter_x, inter_y = x_jp1, y_jp1
                        scan_len.append(np.sqrt(inter_x ** 2 + inter_y ** 2))
                    else:
                        if new_angle_j == angle_i:
                            scan_len.append(np.sqrt(x_j ** 2 + y_j ** 2))
                        else:
                            scan_len.append(np.sqrt(x_j ** 2 + y_j ** 2))
                if len(scan_len):
                    new_scan.append(np.min(scan_len))
                else:
                    # no intersection found
                    angle_diff = np.abs(new_angles - angle_i)
                    idx1, idx2 = np.argsort(angle_diff)[:2]
                    inter_x, inter_y = find_intersect(x[idx1], y[idx1], x[idx2], y[idx2], angle_i)
                    new_scan.append(np.sqrt(inter_x ** 2 + inter_y ** 2))

            new_scans.append(new_scan)

            # new gp
            new_gp = gp - np.array([trans_x, trans_y])
            new_gp[:4] += np.linspace(1, 0, 5, endpoint=False)[1:, np.newaxis] * np.array([trans_x, trans_y])
            new_gps.append(new_gp)

        return np.array(new_scans), np.array(new_gps)

    def train(self, X_train, y_train, X_test, y_test):
        if not self.mode_inited:
            self._init_model()
            self.mode_inited = True

        scan_train, gp_train = X_train
        scan_test, gp_test = X_test

        if self.translation:
            with open("translation_aug.pkl", "rb") as f:
                d = pickle.load(f)
            (scan_aug_train_full, gp_aug_train_full), y_aug_train_full = d['X_train'], d['y_train']
            (scan_aug_test_full, gp_aug_test_full), y_aug_test_full = d['X_test'], d['y_test']
            full_contexts = ["curve", "open_space", "U_turn", "narrow_entrance", "narrow_corridor",
                             "normal_1", "normal_2"]

            scan_aug_train, gp_aug_train, y_aug_train = [], [], []
            scan_aug_test, gp_aug_test, y_aug_test = [], [], []
            for y, ctx in enumerate(self.used_context):
                y_label = full_contexts.index(ctx)
                scan_aug_train.extend(scan_aug_train_full[y_aug_train_full == y_label])
                gp_aug_train.extend(gp_aug_train_full[y_aug_train_full == y_label])
                y_aug_train.extend([y] * np.sum(y_aug_train_full == y_label))

                scan_aug_test.extend(scan_aug_test_full[y_aug_test_full == y_label])
                gp_aug_test.extend(gp_aug_test_full[y_aug_test_full == y_label])
                y_aug_test.extend([y] * np.sum(y_aug_test_full == y_label))

            scans, gps, ys = shuffle(np.concatenate([scan_aug_train, scan_aug_test]),
                                     np.concatenate([gp_aug_train, gp_aug_test]),
                                     np.concatenate([y_aug_train, y_aug_test]))
            num_train = int(len(scans) * 0.8)
            scan_aug_train, gp_aug_train, y_aug_train = scans[:num_train], gps[:num_train], ys[:num_train]
            scan_aug_test, gp_aug_test, y_aug_test = scans[num_train:], gps[num_train:], ys[num_train:]
            if self.full_train:
                scan_train = np.concatenate([scan_train, scan_aug_train, scan_aug_test], axis=0)
                gp_train = np.concatenate([gp_train, gp_aug_train, gp_aug_test], axis=0)
                y_train = np.concatenate([y_train, y_aug_train, y_aug_test], axis=0)
            else:
                scan_train = np.concatenate([scan_train, scan_aug_train], axis=0)
                gp_train = np.concatenate([gp_train, gp_aug_train], axis=0)
                y_train = np.concatenate([y_train, y_aug_train], axis=0)
                scan_test = np.concatenate([scan_test, scan_aug_test], axis=0)
                gp_test = np.concatenate([gp_test, gp_aug_test], axis=0)
                y_test = np.concatenate([y_test, y_aug_test], axis=0)

        labels, counts = np.unique(y_train, return_counts=True)
        weights = counts / counts.sum() * len(counts)
        weights = dict(zip(*[labels, weights]))

        sess = self.sess
        self.writer.add_graph(sess.graph)
        for i in range(self.epochs):
            train_acc, train_loss, test_acc, test_loss = [], [], [], []
            train_confs = []
            scan_train, gp_train, y_train = shuffle(scan_train, gp_train, y_train)
            for j in range(len(scan_train) // self.batch_size + 1):
                batch_scan = scan_train[j:j + self.batch_size]
                batch_gp = gp_train[j:j + self.batch_size]
                batch_y = y_train[j:j + self.batch_size]
                batch_w = np.array([weights[y] for y in batch_y])

                batch_scan, batch_gp, batch_valid = self._data_augment(batch_scan, batch_gp, training=True)
                batch_scan, batch_gp, batch_y, batch_w = \
                    batch_scan[batch_valid], batch_gp[batch_valid], batch_y[batch_valid], batch_w[batch_valid]
                loss, acc, _ = sess.run([self.loss, self.acc, self.train_op],
                                        feed_dict={self.scan_ph: batch_scan,
                                                   self.gp_ph: batch_gp,
                                                   self.label_ph: batch_y,
                                                   self.weight_ph: batch_w,
                                                   self.annealing_step_ph: 1 * (len(X_train) // self.batch_size + 1),
                                                   self.dropout_rate_ph: self.dropout_rate})
                if self.use_EDL:
                    conf = sess.run(self.confidence,
                                    feed_dict={self.scan_ph: batch_scan,
                                               self.gp_ph: batch_gp,
                                               self.label_ph: batch_y,
                                               self.annealing_step_ph: 1 * (len(X_train) // self.batch_size + 1),
                                               self.dropout_rate_ph: 0.0})
                    train_confs.extend(conf)
                train_loss.extend([loss] * self.batch_size)
                train_acc.extend([acc] * self.batch_size)

            test_pred = []
            test_confs = []
            for j in range(0, len(scan_test), self.batch_size):
                batch_scan = scan_test[j:j + self.batch_size]
                batch_gp = gp_test[j:j + self.batch_size]
                batch_y = y_test[j:j + self.batch_size]
                batch_w = np.array([weights[y] for y in batch_y])

                batch_scan, batch_gp, batch_valid = self._data_augment(batch_scan, batch_gp, training=False)
                # batch_scan, batch_gp, batch_y = batch_scan[batch_valid], batch_gp[batch_valid], batch_y[batch_valid]
                pred, loss, acc = sess.run([self.pred, self.loss, self.acc],
                                           feed_dict={self.scan_ph: batch_scan,
                                                      self.gp_ph: batch_gp,
                                                      self.label_ph: batch_y,
                                                      self.weight_ph: batch_w,
                                                      self.annealing_step_ph: 2 ** 31 - 1,
                                                      self.dropout_rate_ph: 0.0})
                if self.use_EDL:
                    conf = sess.run(self.confidence, feed_dict={self.scan_ph: batch_scan,
                                                                self.gp_ph: batch_gp,
                                                                self.label_ph: batch_y,
                                                                self.annealing_step_ph: 2 ** 31 - 1,
                                                                self.dropout_rate_ph: 0.0})
                    test_confs.extend(conf)
                test_loss.extend([loss] * len(batch_scan))
                test_acc.extend([acc] * len(batch_scan))
                test_pred.extend(pred)
            test_pred = np.array(test_pred)
            test_confs = np.array(test_confs)

            train_acc, train_loss = np.mean(train_acc), np.mean(train_loss)
            test_acc, test_loss = np.mean(test_acc), np.mean(test_loss)

            summary = tf.Summary(value=[tf.Summary.Value(tag="train/loss", simple_value=train_loss),
                                        tf.Summary.Value(tag="train/acc", simple_value=train_acc),
                                        tf.Summary.Value(tag="test/loss", simple_value=test_loss),
                                        tf.Summary.Value(tag="test/acc", simple_value=test_acc)])
            self.writer.add_summary(summary, i)
            if self.use_EDL:
                summary = tf.Summary(value=[tf.Summary.Value(tag="train/conf", simple_value=np.mean(train_confs)),
                                            tf.Summary.Value(tag="test/conf", simple_value=np.mean(test_confs))])
                self.writer.add_summary(summary, i)

            summary_val = []
            for j in range(self.Dy):
                acc = np.mean(y_test[y_test == j] == test_pred[y_test == j])
                summary_val.append(tf.Summary.Value(tag="test_acc/{}".format(j), simple_value=acc))
                if self.use_EDL:
                    conf = np.mean(test_confs[y_test == j])
                    summary_val.append(tf.Summary.Value(tag="test_conf/{}".format(j), simple_value=conf))
            self.writer.add_summary(tf.Summary(value=summary_val), i)

            if (i + 1) % 10 == 0:
                self._save_model(epoch_num=i + 1)

    def predict(self, scan, gp):
        in_batch = len(scan.shape) > 1
        if not in_batch:
            scan, gp = np.array([scan]), np.array([gp])
        scan, gp, valid = self._data_augment(scan, gp, training=False)
        if self.use_EDL:
            pred, confidence = self.sess.run([self.pred, self.confidence], feed_dict={self.scan_ph: scan,
                                                                                      self.gp_ph: gp,
                                                                                      self.dropout_rate_ph: 0.0})
            confidence[np.logical_not(valid)] = 0.0
            if not in_batch:
                pred, confidence = pred[0], confidence[0]
            return pred, confidence
        else:
            pred = self.sess.run(self.pred, feed_dict={self.scan_ph: scan, self.gp_ph: gp, self.dropout_rate_ph: 0.0})
            if not in_batch:
                pred = pred[0]
            return pred


def main():
    args = make_args()
    X_train, y_train, X_test, y_test = get_dataset(args, draw_data=False)
    model = ScanClassifier(args)
    model.train(X_train, y_train, X_test, y_test)

    # model._load_model("../rslts/2020-10-14-15-21-06/models/model_500.pickle")
    # print(np.unique(model.predict(X_test[0][y_test == 4], X_test[1][y_test == 4])[0], return_counts=True))
    # print(np.unique(model.predict(X_train[0][y_train == 4], X_train[1][y_train == 4])[0], return_counts=True))
    # print(np.unique(model.predict(X_test[0][y_test == 5], X_test[1][y_test == 5])[0], return_counts=True))
    # print(np.unique(model.predict(X_train[0][y_train == 5], X_train[1][y_train == 5])[0], return_counts=True))

if __name__ == "__main__":
    main()
