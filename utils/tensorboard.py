import tensorflow as tf

def add_to_tensorboard(metrics: dict, step, prefix):
    for key in metrics:
        if metrics[key] is None:
            return

        if key in ["metrics", "metrics_per_class", "metrics_auc"]:
            for m in metrics[key]:
                tf.summary.scalar('{}/{}'.format(prefix, m.name), m.result().numpy(), step=step)

        if key is "image":
            img = metrics[key][0]
            minimum = tf.reduce_min(img, (0, 1), True)
            maximum = tf.reduce_max(img, (0, 1), True)
            img = (img - minimum) / (maximum - minimum)
            img = tf.expand_dims(img, 0)
            tf.summary.image('image/{}/{}'.format(prefix, key), img, step=step)

