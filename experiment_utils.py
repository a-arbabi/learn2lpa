import evaluation
from sklearn.model_selection import KFold
import numpy as np
import data_tools.data_utils as data_utils


def print_result_for_latex(average_result, std_result, exp_name="--", report_path=None):
    print("Method & " + " & ".join([measure for measure in average_result]) + " \\\\")
    printable_results = (
        exp_name
        + " & "
        + " & ".join(
            [
                "${:.4f} \\pm {:.4f}$".format(
                    average_result[measure], std_result[measure]
                )
                for measure in average_result
            ]
        )
        + "\\\\"
    )
    print(printable_results, flush=True)
    if report_path is not None:
        with open(report_path, "w") as f:
            print(printable_results, flush=True, file=f)


def cross_validation(
    data_path,
    train_and_test_func,
    exp_name="",
    report_path=None,
):
    print("loading datasets...", flush=True)
    graph, labels, fold_ids = data_utils.load_dataset(
        data_path
    )
    print("datasets loaded.", flush=True)
    all_prots = [id for ids in fold_ids.values() for id in ids]
    cv_results = {}
    round_count = 0
    for train_protids, valid_test_protids in KFold(n_splits=5, shuffle=False).split(
        all_prots
    ):
        valid_test_protids = [
            valid_test_protids[: len(valid_test_protids) // 2],
            valid_test_protids[len(valid_test_protids) // 2 :],
        ]
        for i in range(2):
            print("CV round: {}\n".format(round_count + 1))
            round_count += 1
            valid_protids = valid_test_protids[i]
            test_protids = valid_test_protids[i - 1]
            fold_ids = {
                "train": train_protids,
                "valid": valid_protids,
                "test": test_protids,
            }
            model_predictions = train_and_test_func(
                fold_ids=fold_ids,
                graph=graph,
                labels=labels,
            )
            cur_res = evaluation.evaluate(
                model_predictions,
                labels[test_protids],
                bootstrap=False,
            )
            for measure in cur_res:
                if measure not in cv_results:
                    cv_results[measure] = []
                cv_results[measure].append(cur_res[measure])
    average_result = {}
    std_result = {}
    for measure in cv_results:
        average_result[measure] = np.mean(cv_results[measure])
        std_result[measure] = np.std(cv_results[measure])
    print_result_for_latex(average_result, std_result, exp_name, report_path)
    return average_result


def temporal(
    data_path,
    train_and_test_func,
    exp_name="",
    report_path=None,
):
    print("loading datasets...", flush=True)
    graph, labels, fold_ids = data_utils.load_dataset(
        data_path
    )
    print("datasets loaded.", flush=True)
    model_predictions = train_and_test_func(
        graph=graph,
        fold_ids=fold_ids,
        labels=labels,
    )
    average_result, std_result = evaluation.evaluate(
        model_predictions,
        labels[fold_ids["test"]],
        bootstrap=True,
    )
    std_result = {
        measure: (abs(std_result[measure][0]) + abs(std_result[measure][1])) / 2.0
        for measure in std_result
    }
    print_result_for_latex(
        average_result, std_result, exp_name, report_path=report_path
    )
