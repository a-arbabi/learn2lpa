import tensorflow as tf
import evaluation


def loss(predictions, labels):
    labels = tf.tile(labels, [predictions.shape[0] // labels.shape[0], 1])
    predictions /= tf.reduce_sum(predictions, axis=1, keepdims=True)
    labels /= tf.reduce_sum(labels, axis=1, keepdims=True)
    return tf.reduce_mean(tf.losses.kl_divergence(labels, predictions))


def results_to_str(res):
    metric_strs = []
    valid_metrics = [
        "step",
        "train_loss_separate",
        "train_loss_combined",
        "valid_loss_separate",
        "valid_loss_combined",
        "macro_aupr",
    ]
    for metric in valid_metrics:
        if metric in res:
            if "step" in metric:
                metric_strs.append(metric + ": {:03d}".format(res[metric]))
            else:
                metric_strs.append(metric + ": {:.3f}".format(res[metric]))
    return ", ".join(metric_strs)


def train_single_model(
    model,
    labels,
    train_ids,
    valid_ids,
    lr=0.2,
    batch_size=512,
    nepochs=20,
    early_stop=True,
    compare_models_by_loss=False,
    verbose=True,
    steps_per_validation=6,
    validations_per_eval=1,
    separate_loss_per_hop=True,
):
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def valid_step(query_ids, separate_loss_per_hop):
        valid_labels = tf.gather(labels, query_ids, 0)
        predictions = model(query_ids, separate_hop_results=separate_loss_per_hop)
        valid_loss = loss(predictions, valid_labels)
        return predictions, valid_loss

    @tf.function
    def train_step(query_ids):
        train_labels = tf.gather(labels, query_ids, 0)
        with tf.GradientTape() as tape:
            predictions = model(query_ids, separate_hop_results=separate_loss_per_hop)
            train_loss = loss(predictions, train_labels)
        gradients = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return train_loss

    def valid_and_eval(valid_ids_dataset, valid_labels, run_eval=True):
        valid_predictions = []
        valid_loss_separate = tf.keras.metrics.Mean()
        valid_loss_combined = tf.keras.metrics.Mean()
        for batch in valid_ids_dataset:
            _, batch_valid_loss_separate = valid_step(batch, separate_loss_per_hop=True)
            batch_valid_predictions, batch_valid_loss_combined = valid_step(
                batch, separate_loss_per_hop=False
            )
            valid_loss_separate.update_state(batch_valid_loss_separate, batch.shape[0])
            valid_loss_combined.update_state(batch_valid_loss_combined, batch.shape[0])
            valid_predictions.append(batch_valid_predictions)

        valid_predictions = tf.concat(valid_predictions, 0)
        out_dict = {
            "valid_loss_separate": valid_loss_separate.result(),
            "valid_loss_combined": valid_loss_combined.result(),
        }
        if run_eval:
            out_dict["macro_aupr"] = evaluation.evaluate(
                valid_predictions.numpy(), valid_labels, verbose=False
            )["Macro AUPR"]
        return out_dict

    train_ids_dataset = (
        tf.data.Dataset.from_tensor_slices(train_ids)
        .shuffle(5000, seed=123)
        .batch(batch_size)
    )
    valid_ids_dataset = tf.data.Dataset.from_tensor_slices(valid_ids).batch(batch_size)

    best_model_weights = None
    best_eval_score = None
    m = tf.keras.metrics.Mean()
    ct_validation_steps = 0

    valid_res = valid_and_eval(valid_ids_dataset, labels[valid_ids])
    valid_res["step"] = 0
    print(results_to_str(valid_res))

    for step_i, batch in enumerate(train_ids_dataset.repeat(nepochs)):
        batch_train_loss = train_step(batch)
        m.update_state(batch_train_loss, batch.shape[0])
        if (step_i + 1) % steps_per_validation == 0:
            step_res = {"step": step_i + 1}
            ct_validation_steps += 1
            step_res["train_loss_separate"] = m.result()
            m.reset_state()
            saved_model_at_this_epoch = False
            run_eval = (
                validations_per_eval is not None
                and (ct_validation_steps % validations_per_eval) == 0
            )
            valid_res = valid_and_eval(valid_ids_dataset, labels[valid_ids], run_eval)
            step_res.update(valid_res)
            print_str = results_to_str(step_res)

            comp_score = None
            if compare_models_by_loss:
                comp_score = -valid_res["valid_loss_combined"]
            else:
                if "macro_aupr" in valid_res:
                    comp_score = valid_res["macro_aupr"]
            if (
                early_stop
                and comp_score is not None
                and (best_eval_score is None or best_eval_score < comp_score)
            ):
                best_eval_score = comp_score
                best_model_weights = model.get_weights()
                saved_model_at_this_epoch = True
            if saved_model_at_this_epoch:
                print_str += " (saved model)"
            if verbose:
                print(print_str, flush=True)

    if best_model_weights is not None:
        model.set_weights(best_model_weights)
