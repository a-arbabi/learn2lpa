import numpy as np
import tensorflow as tf
import models

def run_classic_lp(graph, train_labels, train_ids, alpha=0.001, cutoff=0.):
    mask = np.float32(graph > cutoff)
    graph = graph*mask
    normalized_g = models.normalize_graph_by_degrees(graph)
    ff = tf.eye(normalized_g.shape[0])-alpha*normalized_g
    ff = tf.linalg.inv(ff)
    ff = tf.linalg.set_diag(
        ff, tf.zeros_like(tf.linalg.diag_part(ff))
    )
    ff = tf.gather(ff, train_ids, axis=1)
    ff = tf.math.divide_no_nan(ff, tf.reduce_sum(ff, axis=1, keepdims=True))


    final_y = tf.matmul(ff, train_labels).numpy()

    average_case = np.mean(train_labels, 0)
    no_ppi_prot_ids = np.nonzero(np.sum(graph, 1)==0)[0]
    final_y[no_ppi_prot_ids] = average_case
    return final_y