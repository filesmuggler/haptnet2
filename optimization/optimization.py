import tensorflow as tf

def update_metrics(y,y_true,metrics:list):
    current_values = list()
    for m in metrics:
        m.update_state(y, y_true)
        current_values.append(m.result().numpy())
    return metrics, current_values

def reset_metrics(metrics: list):
    for m in metrics:
        m.reset_states()
    return metrics

def create_confusion_metrics(num_classes, top_k):
    return [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.Precision(name='precision', top_k=top_k),
        tf.keras.metrics.Recall(name='recall', top_k=top_k),
        tf.keras.metrics.MeanIoU(num_classes=num_classes, name='meanIOU')
    ]

def create_accuracy_metrics(class_names):
    return [tf.keras.metrics.SparseCategoricalAccuracy(name='acc_{}'.format(class_names[i])) for i in
            range(len(class_names))]

def create_auc_metrics(class_names):
    return [tf.keras.metrics.AUC(name='auc_{}'.format(class_names[i])) for i in range(len(class_names))]

def check_best_metric(metric, current_best, print_best, step, metrics_to_display: list = None):
    save_model = False
    if current_best is not None:
        if metric.result().numpy() > current_best:
            save_model = True
            current_best = metric.result().numpy()

            if print_best:
                print("Step {}. Current best metric: {}".format(step, current_best))
                if metrics_to_display is not None:
                    [print("{}: {}".format(m.name, m.result().numpy())) for m in metrics_to_display]

    return save_model, current_best