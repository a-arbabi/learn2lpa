from sklearn.metrics import auc
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


@tf.function
def get_recall(labels, predictions, macro_average):
    if macro_average:
        axis = [1]
    else:
        axis = [0, 1]
    total_relevant = tf.reduce_sum(labels, axis)
    true_positives = tf.reduce_sum(predictions * labels, axis)
    recalls = true_positives / total_relevant
    return tf.reduce_mean(recalls)


@tf.function
def get_precision(labels, predictions, macro_average):
    if macro_average:
        axis = [1]
    else:
        axis = [0, 1]
    total_called = tf.reduce_sum(predictions, axis)  # , keepdims=True)
    true_positives = tf.reduce_sum(predictions * labels, axis)
    if tf.reduce_sum(total_called) == 0:
        return 1.0
    if macro_average:
        precisions = tf.math.divide_no_nan(
            true_positives,
            total_called,
        )
        valid_rows = tf.reduce_sum(tf.cast(total_called > 0, tf.float32))
        precision = tf.reduce_sum(precisions) / valid_rows
    else:
        precision = true_positives / total_called
    return precision


def get_aupr(labels, predictions, thresholds, macro_average, plot_curve=False):
    fmeasures = []
    recalls_list = []
    precisions_list = []
    thresholds = np.array(thresholds)
    for cutoff in thresholds:
        hits = tf.cast(predictions >= cutoff, tf.float32)
        recall = get_recall(labels, hits, macro_average=macro_average)
        precision = get_precision(labels, hits, macro_average=macro_average)
        fmeasure = 2.0 * precision * recall / (precision + recall)
        fmeasures.append(fmeasure)
        precisions_list.append(precision)
        recalls_list.append(recall)
    fmeaure_max_arg = np.argmax(fmeasures)
    fmax = np.max(fmeasures)
    aupr = auc(recalls_list, precisions_list)
    if plot_curve:
        plt.plot(
            recalls_list,
            precisions_list,
            label="AUC = %0.3f" % aupr,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Fmax (Macro)")
        plt.plot(
            recalls_list[fmeaure_max_arg],
            precisions_list[fmeaure_max_arg],
            "ro",
            label="Fmax = %0.3f" % fmax,
        )
        plt.legend(loc="upper right")
        plt.show()
    return {"fmax": fmax, "aupr": aupr}


def get_term_centric_auroc(labels, predictions, thresholds):
    total_pos_per_term = tf.reduce_sum(labels, 0)
    total_neg_per_term = tf.reduce_sum(1 - labels, 0)
    tprs = []
    fprs = []
    for cutoff in thresholds:
        preds = tf.cast(predictions >= cutoff, tf.float32)
        tpr = tf.math.divide_no_nan(
            tf.reduce_sum(preds * labels, 0), total_pos_per_term
        )
        fpr = tf.math.divide_no_nan(
            tf.reduce_sum(preds * (1 - labels), 0), total_neg_per_term
        )
        tprs.append(tpr)
        fprs.append(fpr)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    all_aucs = np.array(
        [auc(fprs[:, term_i], tprs[:, term_i]) for term_i in range(labels.shape[1])]
    )

    valid_terms_mask = (total_pos_per_term.numpy() > 0) * (
        total_neg_per_term.numpy() > 0
    )
    auroc = np.sum(all_aucs * valid_terms_mask) / np.sum(valid_terms_mask)
    return auroc


def get_accuracy(
    predicted_scores,
    labels,
):
    predictions = np.argmax(predicted_scores, 1)
    labels = np.argmax(labels, 1)
    return np.mean(predictions == labels)


def evaluate(
    predicted_scores,
    labels,
    verbose=True,
    plot_curve=False,
    bootstrap=False,
    only_macro=False,
):
    if bootstrap:
        eval_prot_ids = range(predicted_scores.shape[0])
        statistics = {}
        confidence = {}
        for i in range(20):
            new_batch = random.choices(eval_prot_ids, k=len(eval_prot_ids))
            batch_res = evaluate(
                predicted_scores[new_batch],
                labels[new_batch],
                verbose=False,
                plot_curve=False,
                bootstrap=False,
            )
            for metric in batch_res:
                if metric not in statistics:
                    statistics[metric] = []
                statistics[metric].append(batch_res[metric])
        for metric in statistics:
            statistics[metric] = sorted(statistics[metric])
            confidence[metric] = [
                np.percentile(statistics[metric], 5),
                np.percentile(statistics[metric], 95),
            ]

    ncutoffs = 100
    cutoff_scores = predicted_scores[np.nonzero(labels)]
    cutoff_scores = sorted(list(set([0] + [x for x in cutoff_scores])))
    cutoffs = (
        [0]
        + [
            cutoff_scores[i]
            for i in range(
                0, len(cutoff_scores), max(1, int(len(cutoff_scores) / ncutoffs))
            )
        ]
        + cutoff_scores[-1:]
    )
    cutoffs += [max(cutoff_scores) + 1.0]
    cutoffs = sorted(list(set(cutoffs)))
    macro_res = get_aupr(
        labels, predicted_scores, cutoffs, macro_average=True, plot_curve=plot_curve
    )
    if not only_macro:
        micro_res = get_aupr(
            labels,
            predicted_scores,
            cutoffs,
            macro_average=False,
            plot_curve=plot_curve,
        )
        term_centric_auroc = get_term_centric_auroc(labels, predicted_scores, cutoffs)
    results = {
        "Macro Fmax": macro_res["fmax"],
        "Macro AUPR": macro_res["aupr"],
    }
    if not only_macro:
        results.update(
            {
                "Micro Fmax": micro_res["fmax"],
                "Micro AUPR": micro_res["aupr"],
                "Term-centric AUROC": term_centric_auroc,
            }
        )
    if bootstrap:
        for metric in results:
            confidence[metric][0] -= results[metric]
            confidence[metric][1] -= results[metric]
    if verbose:
        if not bootstrap:
            printable_results = "\t".join(
                ["{}: {:.4f}".format(measure, results[measure]) for measure in results]
            )
        else:
            printable_results = "\t".join(
                [
                    "{}: {:.4f} ({:.4f}, {:.4f})".format(
                        measure,
                        results[measure],
                        confidence[measure][0],
                        confidence[measure][1],
                    )
                    for measure in results
                ]
            )
        print(printable_results, flush=True)

    if bootstrap:
        return results, confidence
    return results


def plot_weights(model_a, model_b):
    train_calib = model_a.get_calib_probs().numpy()
    train_average_w = model_a.get_averaging_probs().numpy()[:, 0, 0]  # [:,2]
    valid_calib = model_b.get_calib_probs().numpy()
    valid_average_w = model_b.get_averaging_probs().numpy()[:, 0, 0]
    # train_average_w = model.get_averaging_probs().numpy()[:,53]
    plt.plot(np.arange(0, 1, 0.05), train_calib, label="learned mapping")
    #    plt.plot(np.arange(0,1,0.05), valid_calib, label='valid proteins')
    plt.plot([0, 1], [0, 1], "--", label="uncalibrated mapping")
    plt.xlabel("original link weight")
    plt.ylabel("calibrated link weight")
    plt.legend()
    plt.show()
    plt.bar(np.arange(5) + 1 - 0.2, train_average_w, width=0.2)
    # plt.bar(np.arange(5)+1, valid_average_w, width=0.2, label='valid proteins')
    plt.xlabel("Hop weight")
    # plt.legend()
    plt.show()