import argparse
import train_utils
import experiment_utils
import models


def none_or_float(value):
    if value == "None":
        return None
    return float(value)


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Run experiment.")
    parser.add_argument("data_dir", help="Path to the data directory.")
    parser.add_argument("experiment_type", choices=["cv", "temporal"])
    parser.add_argument("--nhops", default=3, type=int)
    parser.add_argument("--nbins", default=20, type=int)
    parser.add_argument("--npows", default=5, type=int)
    parser.add_argument("--calib_before_lp_method", default="local", type=str)
    parser.add_argument("--calib_after_lp_method", default="global", type=str)
    parser.add_argument("--learn_node_importance", default=True, type=str2bool)
    parser.add_argument("--normalize_by_degrees", default=False, type=str2bool)
    parser.add_argument("--use_cumsum", default=False, type=str2bool)
    parser.add_argument("--lr", default=0.2, type=float)
    parser.add_argument("--nepochs", default=20, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--compare_models_by_loss", default=False, type=str2bool)
    parser.add_argument("--separate_loss_per_hop", default=True, type=str2bool)
    parser.add_argument("--exp_name", default="new experiment", type=str)
    parser.add_argument("--report_path", default=None, type=str)
    args = parser.parse_args()
    print(args)

    model_params = {
        "nbins": args.nbins,
        "npows": args.npows,
        "calib_before_lp_method": args.calib_before_lp_method,
        "calib_after_lp_method": args.calib_after_lp_method,
        "learn_node_importance": args.learn_node_importance,
        "normalize_by_degrees": args.normalize_by_degrees,
        "use_cumsum": args.use_cumsum,
        "nhops": args.nhops,
    }
    training_params = {
        "nepochs": args.nepochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "compare_models_by_loss": args.compare_models_by_loss,
        "separate_loss_per_hop": args.separate_loss_per_hop,
        "steps_per_validation": 6,
        "validations_per_eval": 1,
    }

    def train_and_test_func(fold_ids, graph, labels):
        model_params.update(
            {
                "original_graph": graph,
                "seed_labels": labels[fold_ids["train"]],
                "seed_ids": fold_ids["train"],
            }
        )
        training_params.update(
            {
                "labels": labels,
                "train_ids": fold_ids["train"],
                "valid_ids": fold_ids["valid"],
            }
        )
        model = models.LPAdaptor(**model_params)
        training_params["model"] = model
        train_utils.train_single_model(**training_params)

        return model(fold_ids["test"], separate_hop_results=False).numpy()

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