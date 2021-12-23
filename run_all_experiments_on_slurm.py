import subprocess
import argparse
from pathlib import Path


def submit_experiment_on_slurm(python_script_name, data_dir, experiment_type, argdict):
    command = "sbatch slurm_job {} {} {}".format(
        python_script_name, data_dir, experiment_type
    )
    argstring = " ".join(
        ["--{} {}".format(argname, str(argdict[argname])) for argname in argdict]
    )
    command = command + " " + argstring
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    del output, error


def ablation(experiment_type, report_path, data_dir):
    main_argdict = {
        "nepochs": 20,
        "lr": 0.2,
        "batch_size": 512,
        "compare_models_by_loss": False,
        "nbins": 20,
        "npows": 5,
        "nhops": 3,
        "calib_before_lp_method": "local",
        "calib_after_lp_method": "global",
        "learn_node_importance": True,
        "normalize_by_degrees": False,
        "use_cumsum": False,
    }
    report_path_for_type = Path(report_path) / experiment_type
    report_path_for_type.mkdir(parents=True, exist_ok=True)
    for argdict_changes in [
        {
            "exp_name": "Ours_(nhops_1)",
            "nhops" :1,
        },
        {
            "exp_name": "Ours_(nhops_2)",
            "nhops" :2,
        },
        {
            "exp_name": "Ours_(nhops_3)",
            "nhops" :3,
        },
        {
            "exp_name": "Ours_(nhops_4)",
            "nhops" :4,
        },
        {
            "exp_name": "Ours_(nhops_5)",
            "nhops" :5,
        },
        {
            "exp_name": "Ours_(only_calib_1)",
            "calib_after_lp_method": "none",
        },
        {
            "exp_name": "Ours_(no_calib)",
            "calib_after_lp_method": "none",
            "calib_before_lp_method": "none",
        },
        {
            "exp_name": "Ours_(both_global)",
            "calib_after_lp_method": "global",
            "calib_before_lp_method": "global",
        },
        {
            "exp_name": "Ours_(no_node_iw)",
            "learn_node_importance": False,
            "normalize_by_degrees": True,
        },
        {
            "exp_name": "Ours_(merged_loss)",
            "separate_loss_per_hop": False,
        },
    ]:
        argdict = {argname: main_argdict[argname] for argname in main_argdict}
        argdict.update(argdict_changes)
        argdict["report_path"] = report_path_for_type / "{}.txt".format(
            argdict["exp_name"]
        )
        submit_experiment_on_slurm(
            python_script_name="run_main_experiment.py",
            data_dir=data_dir,
            experiment_type=experiment_type,
            argdict=argdict,
        )


def baselines(experiment_type, report_path, data_dir):
    report_path_for_type = Path(report_path) / experiment_type
    report_path_for_type.mkdir(parents=True, exist_ok=True)
    for argdict in [
        {
            "exp_name": "classic_lp",
            "model": "classic_lp",
            "alpha": 0.001,
            "cutoff": 0.0,
        },
        {
            "exp_name": "hpolabeler_lr",
            "model": "hpolabeler_lr",
        },
        {
            "exp_name": "hpolabeler_nn",
            "model": "hpolabeler_nn",
        },
    ]:
        argdict["report_path"] = report_path_for_type / "{}.txt".format(
            argdict["exp_name"]
        )
        submit_experiment_on_slurm(
            python_script_name="run_baseline_experiment.py",
            data_dir=data_dir,
            experiment_type=experiment_type,
            argdict=argdict,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download data for protein function prediction."
    )
    parser.add_argument("report_path", help="Path to the report directory.")
    parser.add_argument("data_dir", help="Path to the data directory.")
    args = parser.parse_args()
    data_path = Path(args.data_dir)
    for experiment_type in ["temporal", "cv"]:
        ablation(experiment_type, args.report_path, data_path / f"ppi_hpo_string_{experiment_type}/")
        baselines(experiment_type, args.report_path, data_path / f"ppi_hpo_string_{experiment_type}/")

if __name__ == "__main__":
    main()