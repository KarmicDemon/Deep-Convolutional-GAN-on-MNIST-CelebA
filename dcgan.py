from matplotlib import pyplot
from sys import exit
import numpy as np
import tensorflow as tf

class DCGAN(object):
    def __init__(self, batch_size = 25, z_dim = 100, learning_rate = .01,
        beta1 = .5, dataset = 'mnist', epochs = 3):
        self.batch_size = batch_size
        self.beta1 = beta1
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gen_count = 0
        self.z_dim = z_dim

        if (dataset != 'mnist' or dataset != 'celeba'):
            print('Unsupported Dataset')
            exit()
        else:
            self.dataset = dataset

        if (self.dataset == 'celeba'):
            self.batch_size = None
            self.beta1 = None
            self.learning_rate = None
            self.z_dim = None

    def discriminator(self, images, reuse = False, alpha = 0.2):
        with tf.variable_scope('discriminator', reuse = reuse):
            disc = tf.layers.conv2d(images, 128, 5, 2, padding = 'same')
            disc = tf.maximum(alpha * disc, disc)

            disc = tf.layers.conv2d(images, 256, 5, 2, padding = 'same')
            disc = tf.layers.batch_normalization(disc, training = True)
            disc = tf.maximum(alpha * disc, disc)

            disc = tf.layers.conv2d(images, 512, 5, 2, padding = 'same')
            disc = tf.layers.batch_normalization(disc, training = True)
            disc = tf.maximum(alpha * disc, disc)

            disc = tf.reshape(disc, (-1, 4 * 4 * 512))
            disc = tf.layers.dense(disc, 1)
            sig_out = tf.sigmoid(disc)

        return (sig_out, disc)

    def generator(self, z, out_channel_dim, is_train = True, alpha = 0.2):
        with tf.variable_scope('generator', reuse = (not is_train)):
            gen = tf.layers.dense(z, 7 * 7 * 512)

            gen = tf.reshape(gen, [-1, 7, 7, 512])
            gen = tf.layers.batch_normalization(gen, training = is_train)
            gen = tf.maximum(alpha * gen, gen)

            gen = tf.layers.conv2d_transpose(gen, 256, 5, 2, padding = 'same')
            gen = tf.layers.batch_normalization(gen, training = is_train)
            gen = tf.maximum(alpha * gen, gen)

            gen = tf.layers.conv2d_transpose(gen, 128, 5, 2, padding = 'same')
            gen = tf.layers.batch_normalization(gen, training = is_train)
            gen = tf.maximum(alpha * gen, gen)

            gen = tf.layers.conv2d_transpose(gen, out_channel_dim, 5, 1,
                padding='same')
            gen = tf.tanh(gen)
        return gen

    def model_loss(self, input_real, input_z, out_channel_dim,
        alpha = 0.2, smooth = 0.1):
        gen = generator(input_z, out_channel_dim)

        real_disc, real_logits = discriminator(input_real)
        fake_disc, fake_logits = discriminator(gen, reuse = True)

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            logits = real_logits, labels = tf.ones_like(real_disc) * \
            (1 - smooth)))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
        logits = fake_logits, labels = tf.zeros_like(fake_disc)))
        gen_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
            logits = fake_logits, labels = tf.ones_like(fake_disc)))

        return (real_loss + fake_loss), gen_loss

    def model_optimization(self, d_loss, g_loss, learning_rate, beta1):
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        with tf.control_dependencies(tf.get_collection(\
            tf.GraphKeys.UPDATE_OPS)):
            d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)\
                .minimize(d_loss, var_list = d_vars)
            g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)\
                .minimize(g_loss, var_list=  g_vars)

        return d_opt, g_opt

    def return_inputs(self, image_width, image_height, image_channels, z_dim):
        real_input = tf.placeholder(tf.float32,
            shape = [None, image_width, image_height, image_channels],
            name = 'real_input')
        z_input = tf.placeholder(tf.float32, shape = [None, z_dim],
            name = 'z_input')
        learning_rate = tf.placeholder(tf.float32, shape = [],
            name = 'learning_rate')

        return (real_input, z_input, learning_rate)

    def save_generator_output(self, n_images, input_z, out_channel_dim,
        image_mode):
        cmap = None if image_mode == 'RGB' else 'gray'

        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

        samples = sess.run(generator(input_z, out_channel_dim, False),
            feed_dict={ input_z: example_z })

        images_grid = helper.images_square_grid(samples, image_mode)
        pyplot.imshow(images_grid, cmap=cmap)
        self.gen_count += 1
        pyplot.savefig('generated_image_' + str(self.gen_count) + '.png',
            bbox_inches = 'tight')
        pyplot.clear()
        pyplot.close()

    def train(self, epoch_count, batch_size, z_dim, learning_rate, beta1,
        get_batches, data_shape, data_image_mode):
        losses = []

        _, image_width, image_height, image_channels = data_shape
        input_real, input_z, lr = model_inputs(image_width,
            image_height, image_channels, z_dim)

        d_loss, g_loss = model_loss(input_real, input_z, image_channels)
        d_train_opt, g_train_opt = model_opt(d_loss, g_loss,
            learning_rate, beta1)

        steps = 0
        with tf.Session() as s:
            s.run(tf.global_variables_initializer())
            for epoch_i in range(epoch_count):
                for batch_images in get_batches(batch_size):
                    steps += 1
                    batch_images = batch_images*2
                    batch_z = np.random.uniform(-1, 1,
                        size = (batch_size, z_dim))

                    _ = sess.run(d_train_opt, feed_dict = {
                                                input_real : batch_images,
                                                input_z : batch_z,
                                                lr : learning_rate })
                    _ = sess.run(g_train_opt, feed_dict = {
                                                input_z : batch_z,
                                                input_real : batch_images,
                                                lr : learning_rate })
                    _ = sess.run(g_train_opt, feed_dict = {
                                                input_z : batch_z,
                                                input_real : batch_images,
                                                lr : learning_rate })

                    if steps % 10==0:
                        train_loss_d = d_loss.eval({
                            input_real : batch_images,
                            input_z : batch_z })
                        train_loss_g = g_loss.eval({ input_z : batch_z })

                        print("Epoch {}/{}...".format(epoch_i+1, epoch_count),
                            "Discriminator Loss: {:.4f}..."
                                .format(train_loss_d),
                            "Generator Loss: {:.4f}".format(train_loss_g))

                        losses.append([train_loss_d, train_loss_g])


                    if steps % 100==0:
                        show_generator_output(sess, 9,
                            input_z, image_channels, data_image_mode)

        return losses

    def start(self):
        self.dataset = helper.Dataset(self.dataset,
            glob(os.path.join(self.dataset + '/*.jpg')))
        with tf.graph().as_default():
            train(self.epochs, self.batch_size, self.z_dim, self.learning_rate,
                self.beta1, dataset.get_batches, dataset.shape,
                dataset.image_mode)
