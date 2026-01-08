#!/usr/bin/env python
from __future__ import print_function
import os, sys, shutil
import argparse
import json, re
from subprocess import check_output
import datetime
import matplotlib.pyplot as plt


def load_json(path):
    f = open(path, "r")
    data = json.load(f)
    return data


def save_json(obj, path):
    tmp_file = path + ".bak"
    f = open(tmp_file, "w")
    json.dump(obj, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
    f.close()
    try:
        os.rename(tmp_file, path)
    except:
        os.remove(path)
        os.rename(tmp_file, path)


def parse_evaluation_result(input_str, i):
    res = {}
    res['frame_number'] = i + 1
    res['error'] = {}
    regex = "([A-Za-z. \\[\\].0-9]+):[ ]*([0-9]*\.[0-9]+|[0-9]+)"
    for elem in re.findall(regex,input_str):
        if "Time" in elem[0]:
            res['time'] = float(elem[1])
        elif "Average" in elem[0]:
            res['error']['average'] = float(elem[1])
        elif "deviation" in elem[0]:
            res['error']['std'] = float(elem[1])
        else:
            res['error'][elem[0]] = float(elem[1])
    return res


def evaluate_sequence(sequence, algorithm, dataset, executable, img_files, gt_files,
                      state, state_path):
    if "eval_results" not in state[dataset][algorithm][-1].keys():
        state[dataset][algorithm][-1]["eval_results"] = {}
    elif sequence in state[dataset][algorithm][-1]["eval_results"].keys():
        return

    res = []
    for i in range(len(img_files) - 1):
        sys.stdout.write("Algorithm: %-20s Sequence: %-10s Done: [%3d/%3d]\r" %
                         (algorithm, sequence, i, len(img_files) - 1)),
        sys.stdout.flush()

        res_string = check_output([executable, img_files[i], img_files[i + 1],
                                   algorithm, gt_files[i]])
        res.append(parse_evaluation_result(res_string, i))
    state[dataset][algorithm][-1]["eval_results"][sequence] = res
    save_json(state, state_path)

#############################DATSET DEFINITIONS################################

def evaluate_mpi_sintel(source_dir, algorithm, evaluation_executable, state, state_path):
    evaluation_result = {}
    img_dir = os.path.join(source_dir, 'mpi_sintel', 'training', 'final')
    gt_dir = os.path.join(source_dir, 'mpi_sintel', 'training', 'flow')
    sequences = [f for f in os.listdir(img_dir)
                 if os.path.isdir(os.path.join(img_dir, f))]
    for seq in sequences:
        img_files = sorted([os.path.join(img_dir, seq, f)
                            for f in os.listdir(os.path.join(img_dir, seq))
                            if f.endswith(".png")])
        gt_files = sorted([os.path.join(gt_dir, seq, f)
                           for f in os.listdir(os.path.join(gt_dir, seq))
                           if f.endswith(".flo")])
        evaluation_result[seq] = evaluate_sequence(seq, algorithm, 'mpi_sintel',
            evaluation_executable, img_files, gt_files, state, state_path)
    return evaluation_result


def evaluate_middlebury(source_dir, algorithm, evaluation_executable, state, state_path):
    evaluation_result = {}
    img_dir = os.path.join(source_dir, 'middlebury', 'other-data')
    gt_dir = os.path.join(source_dir, 'middlebury', 'other-gt-flow')
    sequences = [f for f in os.listdir(gt_dir)
                 if os.path.isdir(os.path.join(gt_dir, f))]
    for seq in sequences:
        img_files = sorted([os.path.join(img_dir, seq, f)
                            for f in os.listdir(os.path.join(img_dir, seq))
                            if f.endswith(".png")])
        gt_files = sorted([os.path.join(gt_dir, seq, f)
                           for f in os.listdir(os.path.join(gt_dir, seq))
                           if f.endswith(".flo")])
        evaluation_result[seq] = evaluate_sequence(seq, algorithm, 'middlebury',
            evaluation_executable, img_files, gt_files, state, state_path)
    return evaluation_result


dataset_eval_functions = {
    "mpi_sintel": evaluate_mpi_sintel,
    "middlebury": evaluate_middlebury
}

###############################################################################

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def parse_sequence(input_str):
    if len(input_str) == 0:
        return []
    else:
        return [o.strip() for o in input_str.split(",") if o]


def build_chart(dst_folder, state, dataset):
    fig = plt.figure(figsize=(16, 10))
    markers = ["o", "s", "h", "^", "D"]
    marker_idx = 0
    colors = ["b", "g", "r"]
    color_idx = 0
    for algo in state[dataset].keys():
        for eval_instance in state[dataset][algo]:
            name = algo + "--" + eval_instance["timestamp"]
            average_time = 0.0
            average_error = 0.0
            num_elem = 0
            for seq in eval_instance["eval_results"].keys():
                for frame in eval_instance["eval_results"][seq]:
                    average_time += frame["time"]
                    average_error += frame["error"]["average"]
                    num_elem += 1
            average_time /= num_elem
            average_error /= num_elem

            marker_style = colors[color_idx] + markers[marker_idx]
            color_idx += 1
            if color_idx >= len(colors):
                color_idx = 0
            marker_idx += 1
            if marker_idx >= len(markers):
                marker_idx = 0
            plt.gca().plot([average_time], [average_error],
                           marker_style,
                           markersize=14,
                           label=name)

    plt.gca().set_ylabel('Average Endpoint Error (EPE)', fontsize=20)
    plt.gca().set_xlabel('Average Runtime (seconds per frame)', fontsize=20)
    plt.gca().set_xscale("log")
    plt.gca().set_title('Evaluation on ' + dataset, fontsize=20)

    plt.gca().legend()
    fig.savefig(os.path.join(dst_folder, "evaluation_results_" + dataset + ".png"),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optical flow benchmarking script',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "bin_path",
        default="./optflow-example-optical_flow_evaluation",
        help="Path to the optical flow evaluation executable")
    parser.add_argument(
        "-a",
        "--algorithms",
        metavar="ALGORITHMS",
        default="",
        help=("Comma-separated list of optical-flow algorithms to evaluate "
              "(example: -a farneback,tvl1,deepflow). Note that previously "
              "evaluated algorithms are also included in the output charts"))
    parser.add_argument(
        "-d",
        "--datasets",
        metavar="DATASETS",
        default="mpi_sintel",
        help=("Comma-separated list of datasets for evaluation (currently only "
              "'mpi_sintel' and 'middlebury' are supported)"))
    parser.add_argument(
        "-f",
        "--dataset_folder",
        metavar="DATASET_FOLDER",
        default="./OF_datasets",
        help=("Path to a folder containing datasets. To enable evaluation on "
              "MPI Sintel dataset, please download it using the following links: "
              "http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip and "
              "http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip and "
              "unzip these archives into the 'mpi_sintel' folder. To enable evaluation "
              "on the Middlebury dataset use the following links: "
              "http://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip, "
              "http://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip. "
              "These should be unzipped into 'middlebury' folder"))
    parser.add_argument(
        "-o",
        "--out",
        metavar="OUT_DIR",
        default="./OF_evaluation_results",
        help="Output directory where to store benchmark results")
    parser.add_argument(
        "-s",
        "--state",
        metavar="STATE_JSON",
        default="./OF_evaluation_state.json",
        help=("Path to a json file that stores the current evaluation state and "
              "previous evaluation results"))
    args, other_args = parser.parse_known_args()

    if not os.path.isfile(args.bin_path):
        print("Error: " + args.bin_path + " does not exist")
        sys.exit(1)

    if not os.path.exists(args.dataset_folder):
        print("Error: " + args.dataset_folder + (" does not exist. Please, correctly "
                                                 "specify the -f parameter"))
        sys.exit(1)

    state = {}
    if os.path.isfile(args.state):
        state = load_json(args.state)

    algorithm_list = parse_sequence(args.algorithms)
    dataset_list = parse_sequence(args.datasets)
    for dataset in dataset_list:
        if dataset not in dataset_eval_functions.keys():
            print("Error: unsupported dataset " + dataset)
            sys.exit(1)
        if dataset not in os.listdir(args.dataset_folder):
            print("Error: " + os.path.join(args.dataset_folder, dataset) + (" does not exist. "
                              "Please, download the dataset and follow the naming conventions "
                              "(use -h for more information)"))
            sys.exit(1)

    for dataset in dataset_list:
        if dataset not in state.keys():
            state[dataset] = {}
        for algorithm in algorithm_list:
            if algorithm in state[dataset].keys():
                last_eval_instance = state[dataset][algorithm][-1]
                if "finished" not in last_eval_instance.keys():
                    print(("Continuing an unfinished evaluation of " +
                          algorithm + " started at " + last_eval_instance["timestamp"]))
                else:
                    state[dataset][algorithm].append({"timestamp":
                        datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")})
            else:
                state[dataset][algorithm] = [{"timestamp":
                    datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")}]
            save_json(state, args.state)
            dataset_eval_functions[dataset](args.dataset_folder, algorithm, args.bin_path,
                                            state, args.state)
            state[dataset][algorithm][-1]["finished"] = True
            save_json(state, args.state)
    save_json(state, args.state)

    create_dir(args.out)
    for dataset in dataset_list:
        build_chart(args.out, state, dataset)
