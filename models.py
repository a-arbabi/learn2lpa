import tensorflow as tf
import numpy as np


def apply_prior(predictions, prior_hps):
    out = tf.where(
        tf.not_equal(tf.reduce_sum(predictions, 1, keepdims=True), 0.0),
        predictions,
        prior_hps,
    )
    return out


def normalize_graph_by_degrees(graph):
    diags_left = tf.sqrt(tf.reduce_sum(graph, 1))
    diags_right = tf.sqrt(tf.reduce_sum(graph, 0))
    degree_mat_left = tf.linalg.diag(
        tf.math.divide_no_nan(tf.ones_like(diags_left), diags_left)
    )
    degree_mat_right = tf.linalg.diag(
        tf.math.divide_no_nan(tf.ones_like(diags_right), diags_right)
    )
    return tf.matmul(tf.matmul(degree_mat_left, graph), degree_mat_right)


class NodeImportanceLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        graph=None,
    ):
        super(NodeImportanceLayer, self).__init__()
        self.column_weights = None
        if graph is not None:
            w_init = tf.math.divide_no_nan(1.0, tf.math.sqrt(tf.reduce_sum(graph, 0)))
            w_init = tf.math.log(w_init + 1e-6)
            w_init = w_init - tf.reduce_mean(w_init)
            w_init = w_init.numpy()
            self.column_weights = self.add_weight(
                shape=[1, graph.shape[1]],
                initializer=tf.constant_initializer(w_init),
                dtype=tf.float32,
            )

    def build(self, input_shapes):
        if self.column_weights is None:
            self.column_weights = self.add_weight(
                shape=[1, input_shapes[1]],
                initializer=tf.constant_initializer(np.zeros([1, input_shapes[1]])),
                dtype=tf.float32,
            )

    def call(self, graph, seed_ids=None):
        column_weights = self.column_weights
        if seed_ids is not None:
            column_weights = tf.gather(column_weights, seed_ids, axis=1)
        column_weights = tf.nn.softmax(column_weights, axis=1) * column_weights.shape[1]
        graph = graph * column_weights
        return graph


def normalize_edge_weights(graph):
    mask = tf.cast(graph > 0, tf.float32)
    mean_edge = tf.reduce_sum(graph) / tf.reduce_sum(mask)
    std_edge = graph - mean_edge
    std_edge *= std_edge
    std_edge = tf.reduce_sum(mask * std_edge) / tf.reduce_sum(mask)
    std_edge = tf.math.sqrt(std_edge)

    graph = (graph - mean_edge) / std_edge
    graph = graph * mask
    return graph


class CalibGlobalLayer(tf.keras.layers.Layer):
    def __init__(self, npows=5):
        super(CalibGlobalLayer, self).__init__()
        kernel_init = np.ones([npows, 1])
        self.layer = tf.keras.layers.Dense(
            1, kernel_initializer=tf.constant_initializer(kernel_init)
        )
        self.npows = npows

    def call(self, graph):
        mask = tf.cast(graph > 0, tf.float32)
        graph = normalize_edge_weights(graph)

        multiplier = tf.math.sqrt(graph * graph)

        last = graph
        sqr = tf.math.divide_no_nan(last * tf.math.sqrt(multiplier), multiplier)
        graph_powers = [graph, sqr]
        for _ in range(self.npows - 2):
            last = last * multiplier
            graph_powers.append(last)
        graph = tf.stack(graph_powers, axis=-1)
        graph = self.layer(graph)
        graph = tf.squeeze(graph)

        graph = tf.nn.elu(graph) + 1.0
        graph *= mask
        return graph


class CalibLocalLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        nbins,
        original_range=[0.0, 1.0],
        sigma_div=50,
        use_cumsum=False,
    ):
        super(CalibLocalLayer, self).__init__()
        partitions = np.linspace(original_range[0], original_range[1], nbins)
        self.bases = tf.constant(partitions, dtype=tf.float32)
        tmp = partitions[1:] - partitions[:-1]
        sigmas = np.zeros_like(partitions)
        sigmas[:-1] += tmp
        sigmas[1:] += tmp
        sigmas[1:-1] /= 2
        self.sigmas = sigmas / sigma_div
        self.use_cumsum = use_cumsum

        if use_cumsum:
            w_init = np.zeros(nbins)
        else:
            w_init = partitions
            w_init = np.log(w_init + 1e-6)
            w_init -= np.mean(partitions)
        w_init = tf.constant_initializer(w_init)

        self.calib_w = self.add_weight(
            shape=[nbins],
            initializer=w_init,
            dtype=tf.float32,
        )

    def get_calib_probs(self):
        probs = tf.nn.softmax(self.calib_w)
        if self.use_cumsum:
            probs = tf.cumsum(probs)
        return probs[tf.newaxis, :]

    def call(self, graph):
        mask = tf.cast(graph > 0, tf.float32)
        z = tf.expand_dims(graph, -1) - self.bases
        z = tf.math.exp(-z * z / self.sigmas)
        z = tf.math.divide_no_nan(z, tf.reduce_sum(z, axis=-1, keepdims=True))
        probs = self.get_calib_probs()
        out = z * probs
        out = tf.reduce_sum(out, -1)
        out = out * mask
        return out


class AdaptHopLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        original_graph,
        seed_ids,
        nbins=20,
        npows=5,
        hop=1,
        calib_before_lp_method="local",
        calib_after_lp_method="global",
        learn_node_importance=True,
        normalize_by_degrees=False,
        use_cumsum=False,
    ):
        super(AdaptHopLayer, self).__init__()
        self.original_graph = tf.constant(original_graph)
        self.seed_ids = seed_ids
        self.hop = hop

        self.calib_before_lp_method = calib_before_lp_method
        self.calib_after_lp_method = calib_after_lp_method
        self.learn_node_importance = learn_node_importance
        self.normalize_by_degrees = normalize_by_degrees

        def _get_calib_layer(calib_method):
            if calib_method == "local":
                return CalibLocalLayer(nbins, use_cumsum=use_cumsum)
            elif calib_method == "global":
                return CalibGlobalLayer(npows=npows)
            elif calib_method == "none":
                return None
            else:
                raise ValueError('Unkown calibration method "{}"'.format(calib_method))

        self.calib_before_lp_layer = _get_calib_layer(calib_before_lp_method)
        if self.hop > 1:
            self.calib_after_lp_layer = _get_calib_layer(calib_after_lp_method)
        else:
            self.calib_after_lp_layer = None

        if learn_node_importance:
            self.node_importance_layer = NodeImportanceLayer(original_graph)

    def call(self, query_ids, graph):
        # Calibrate edge weights
        if self.calib_before_lp_layer is not None:
            graph = self.calib_before_lp_layer(graph)

        # Normalize graph matrix by node degrees
        if self.normalize_by_degrees:
            graph = normalize_graph_by_degrees(graph)

        # Apply node importance weights
        if self.learn_node_importance:
            graph = self.node_importance_layer(graph)

        # Propagate labels in the graph
        graph_mult = graph
        graph = tf.gather(graph, query_ids, axis=0)
        for _ in tf.range(self.hop - 1):
            graph = tf.linalg.matmul(
                graph, graph_mult, a_is_sparse=False, b_is_sparse=True
            )

        # Remove self loops (i.e. set matrix diagonal to zero)
        diag_mask = tf.ones_like(self.original_graph)
        diag_mask = tf.linalg.set_diag(
            diag_mask, tf.zeros_like(tf.linalg.diag_part(diag_mask))
        )
        diag_mask = tf.gather(diag_mask, query_ids, axis=0)
        graph = graph * diag_mask

        # Calibrate edge weights
        graph = tf.gather(graph, self.seed_ids, axis=1)
        if self.calib_after_lp_layer is not None:
            graph = self.calib_after_lp_layer(graph)

        graph = tf.math.divide_no_nan(
            graph, tf.reduce_sum(graph, axis=1, keepdims=True)
        )
        return graph


class LPAdaptor(tf.keras.Model):
    def __init__(
        self,
        original_graph,
        seed_labels,
        seed_ids,
        nbins=20,
        npows=5,
        nhops=3,
        calib_before_lp_method="local",
        calib_after_lp_method="global",
        learn_node_importance=True,
        normalize_by_degrees=False,
        use_cumsum=False,
    ):
        super(LPAdaptor, self).__init__()
        self.seed_labels = tf.constant(seed_labels)
        self.prior_label = tf.reduce_mean(seed_labels, 0)
        self.original_graph = tf.constant(original_graph)

        self.adaptors = [
            AdaptHopLayer(
                original_graph=original_graph,
                nbins=nbins,
                npows=npows,
                seed_ids=seed_ids,
                calib_before_lp_method=calib_before_lp_method,
                calib_after_lp_method=calib_after_lp_method,
                learn_node_importance=learn_node_importance,
                normalize_by_degrees=normalize_by_degrees,
                use_cumsum=use_cumsum,
                hop=hop,
            )
            for hop in range(1, nhops + 1)
        ]

    def call(self, query_ids, separate_hop_results):
        graph = self.original_graph
        graph = tf.stack(
            [adaptor(query_ids, graph) for adaptor in self.adaptors], axis=0
        )
        if separate_hop_results:
            graph = tf.reshape(graph, [graph.shape[0] * graph.shape[1], graph.shape[2]])
        else:
            graph = tf.reduce_sum(graph, 0)
            graph = tf.math.divide_no_nan(
                graph, tf.reduce_sum(graph, axis=1, keepdims=True)
            )
        out = tf.matmul(graph, self.seed_labels, a_is_sparse=True, b_is_sparse=True)
        out = apply_prior(out, self.prior_label)
        return out
