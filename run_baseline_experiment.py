import argparse
from bayes_opt import BayesianOptimization
import experiment_utils
import numpy as np
import baseline_models.classic_lp as classic_lp
import evaluation
import pandas as pd
import baseline_models.hpolabeler_base as hpolabeler_base

def build_hpolabeler_lr_train_and_test_func():
    def train_and_test_func(fold_ids, graph, labels):
        train_protids = fold_ids["train"]
        test_protids = fold_ids["test"]

        train_annotation = pd.DataFrame(labels[train_protids])
        train_feature = pd.DataFrame(graph[train_protids])
        test_feature = pd.DataFrame(graph[test_protids])

        classifier = hpolabeler_base.FlatModel(model="lr")
        classifier.fit(train_feature, train_annotation)
        pr = classifier.predict(test_feature)
        pr_np = np.zeros_like(labels)
        for i, protid in enumerate(test_protids):
            pr_np[protid] = np.array([pr[i][j] for j in range(len(pr[i]))])
        return pr_np[test_protids]

    return train_and_test_func


def build_hpolabeler_nn_train_and_test_func():
    def train_and_test_func(fold_ids, graph, labels):#(fold_ids, graph, labels):
        train_protids = fold_ids["train"]
        test_protids = fold_ids["test"]
        ppi_dict = {}
        for i, j in zip(*np.nonzero(graph)):
            if i not in ppi_dict:
                ppi_dict[i] = {}
            ppi_dict[i][j] = graph[i, j]
        train_ant_dict = {
            i: list(np.nonzero(labels[i])[0]) for i in train_protids
        }
        pr = hpolabeler_base.neighbor_scoring(ppi_dict, test_protids, train_ant_dict)
        mean_labels = np.mean(labels[train_protids], 0)
        pr_np = np.zeros_like(labels)
        for i in test_protids:
            if i not in pr:
                pr_np[i] = mean_labels
            else:
                for j in pr[i]:
                    pr_np[i][j] = pr[i][j]
        return pr_np[test_protids]

    return train_and_test_func


def build_classic_lp_train_and_test_func(alpha=None, cutoff=None):
    def train_and_test_func(fold_ids, graph, labels):
        _alpha = alpha
        _cutoff = cutoff
        if alpha is None or cutoff is None:
            pbounds = {'alpha': (0.001, .99), 'cutoff': (0., .99)}
            if alpha is not None:
                pbounds['alpha'] = (alpha, alpha)
            if cutoff is not None:
                pbounds['cutoff'] = (cutoff, cutoff)
            def try_lp(alpha, cutoff):
                out = classic_lp.run_classic_lp(
                    graph, labels[fold_ids["train"]], fold_ids["train"], alpha, cutoff
                )
                return evaluation.evaluate(out[fold_ids['valid']], labels[fold_ids['valid']], verbose=False)['Macro AUPR']

            optimizer = BayesianOptimization(
                f=try_lp,
                pbounds=pbounds,
                random_state=1,
            )
            optimizer.maximize(
                init_points=2,
                n_iter=50,
            )
            _alpha = optimizer.max['params']["alpha"]
            _cutoff = optimizer.max['params']["cutoff"]

        return classic_lp.run_classic_lp(
            graph, labels[fold_ids["train"]], fold_ids["train"], _alpha, _cutoff
        )[fold_ids["test"]]
        

    return train_and_test_func



def main():
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument("data_dir", help="Path to the data directory.")
    parser.add_argument("experiment_type", choices=["cv", "temporal"])
    parser.add_argument("--model", default="--", type=str)
    parser.add_argument("--exp_name", default="baseline experiment", type=str)
    parser.add_argument("--report_path", default=None, type=str)
    parser.add_argument("--alpha", default=0.001, type=float)
    parser.add_argument("--cutoff", default=0.0, type=float)
    args = parser.parse_args()
    print(args)

    if args.model == "classic_lp":
        train_and_test_func = build_classic_lp_train_and_test_func(
            alpha=args.alpha,
            cutoff=args.cutoff,
        )
    elif args.model == "hpolabeler_nn":
        train_and_test_func = build_hpolabeler_nn_train_and_test_func()
    elif args.model == "hpolabeler_lr":
        train_and_test_func = build_hpolabeler_lr_train_and_test_func()


    print("experiment: {}".format(args.experiment_type))
    if args.experiment_type == "cv":
        experiment_utils.cross_validation(
            args.data_dir,
            train_and_test_func,
            args.exp_name,
            report_path=args.report_path,
        )
    elif args.experiment_type == "temporal":
        experiment_utils.temporal(
            args.data_dir,
            train_and_test_func,
            args.exp_name,
            report_path=args.report_path,
        )


if __name__ == "__main__":
    main()