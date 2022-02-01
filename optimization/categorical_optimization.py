import time
import tensorflow as tf
from tqdm import tqdm

from optimization import *
from utils.tensorboard import *

def train_classification(model,writer,dataset,optimizer,previous_steps,num_classes):
    #metrics
    metric_loss = tf.keras.metrics.Mean(name="MeanLoss")
    # auc_metric = create_auc_metrics()
    ca_metric_loss = tf.keras.metrics.CategoricalAccuracy(name="CatAcc")
    confusion_metrics = create_confusion_metrics(num_classes, top_k=5)

    for x_train, y_train in tqdm(dataset):
        with tf.GradientTape() as tape:
            logits_y = model(x_train, training=True)

            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

            # compute loss
            loss_value = tf.reduce_mean(loss_fn(y_train,logits_y))

            ca_metric_loss.update_state(logits_y,y_train)
            metric_loss.update_state(loss_value.numpy())

            # add regularization
            vars = model.trainable_variables
            l2_reg = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars]) * 0.001
            loss_value += l2_reg

        # apply gradients
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        update_metrics(logits_y, y_train, confusion_metrics, 0.5)

        with writer.as_default():
            add_to_tensorboard({
                "metrics": [metric_loss, ca_metric_loss, *confusion_metrics]
                #"full_tensor": [confusion_matrix]
            }, previous_steps, "train")
            writer.flush()

        # TODO: should it be reset after epoch not after batch?
        previous_steps += 1
        metric_loss.reset_states()
        ca_metric_loss.reset_states()
        confusion_matrix.reset_states()
        reset_metrics(confusion_metrics)

    return previous_steps