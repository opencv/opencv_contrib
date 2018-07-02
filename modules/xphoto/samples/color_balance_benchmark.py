#!/usr/bin/env python
from __future__ import print_function
import os, sys, argparse, json
import numpy as np
import scipy.io
import cv2 as cv
import timeit
from learn_color_balance import load_ground_truth


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


def parse_sequence(input_str):
    if len(input_str) == 0:
        return []
    else:
        return [o.strip() for o in input_str.split(",") if o]


def stretch_to_8bit(arr, clip_percentile = 2.5):
    arr = np.clip(arr * (255.0 / np.percentile(arr, 100 - clip_percentile)), 0, 255)
    return arr.astype(np.uint8)


def evaluate(im, algo, gt_illuminant, i, range_thresh, bin_num, dst_folder, model_folder):
    new_im = None
    start_time = timeit.default_timer()
    if algo=="grayworld":
        inst = cv.xphoto.createGrayworldWB()
        inst.setSaturationThreshold(0.95)
        new_im = inst.balanceWhite(im)
    elif algo=="nothing":
        new_im = im
    elif algo.split(":")[0]=="learning_based":
        model_path = ""
        if len(algo.split(":"))>1:
            model_path = os.path.join(model_folder, algo.split(":")[1])
        inst = cv.xphoto.createLearningBasedWB(model_path)
        inst.setRangeMaxVal(range_thresh)
        inst.setSaturationThreshold(0.98)
        inst.setHistBinNum(bin_num)
        new_im = inst.balanceWhite(im)
    elif algo=="GT":
        gains = gt_illuminant / min(gt_illuminant)
        g1 = float(1.0 / gains[2])
        g2 = float(1.0 / gains[1])
        g3 = float(1.0 / gains[0])
        new_im = cv.xphoto.applyChannelGains(im, g1, g2, g3)
    time = 1000*(timeit.default_timer() - start_time) #time in ms

    if len(dst_folder)>0:
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        im_name = ("%04d_" % i) + algo.replace(":","_") + ".jpg"
        cv.imwrite(os.path.join(dst_folder, im_name), stretch_to_8bit(new_im))

    #recover the illuminant from the color balancing result, assuming the standard model:
    estimated_illuminant = [0, 0, 0]
    eps = 0.01
    estimated_illuminant[2] = np.percentile((im[:,:,0] + eps) / (new_im[:,:,0] + eps), 50)
    estimated_illuminant[1] = np.percentile((im[:,:,1] + eps) / (new_im[:,:,1] + eps), 50)
    estimated_illuminant[0] = np.percentile((im[:,:,2] + eps) / (new_im[:,:,2] + eps), 50)

    res = np.arccos(np.dot(gt_illuminant,estimated_illuminant)/
                   (np.linalg.norm(gt_illuminant) * np.linalg.norm(estimated_illuminant)))
    return (time, (res / np.pi) * 180)


def build_html_table(out, state, stat_list, img_range):
    stat_dict = {'mean': ('Mean error', lambda arr: np.mean(arr)),
                 'median': ('Median error',lambda arr: np.percentile(arr, 50)),
                 'p05': ('5<sup>th</sup> percentile',lambda arr: np.percentile(arr, 5)),
                 'p20': ('20<sup>th</sup> percentile',lambda arr: np.percentile(arr, 20)),
                 'p80': ('80<sup>th</sup> percentile',lambda arr: np.percentile(arr, 80)),
                 'p95': ('95<sup>th</sup> percentile',lambda arr: np.percentile(arr, 95))
                }
    html_out = ['<style type="text/css">\n',
                '  html, body {font-family: Lucida Console, Courier New, Courier;font-size: 16px;color:#3e4758;}\n',
                '  .tbl{background:none repeat scroll 0 0 #FFFFFF;border-collapse:collapse;font-family:"Lucida Sans Unicode","Lucida Grande",Sans-Serif;font-size:14px;margin:20px;text-align:left;width:480px;margin-left: auto;margin-right: auto;white-space:nowrap;}\n',
                '  .tbl span{display:block;white-space:nowrap;}\n',
                '  .tbl thead tr:last-child th {padding-bottom:5px;}\n',
                '  .tbl tbody tr:first-child td {border-top:3px solid #6678B1;}\n',
                '  .tbl th{border:none;color:#003399;font-size:16px;font-weight:normal;white-space:nowrap;padding:3px 10px;}\n',
                '  .tbl td{border:none;border-bottom:1px solid #CCCCCC;color:#666699;padding:6px 8px;white-space:nowrap;}\n',
                '  .tbl tbody tr:hover td{color:#000099;}\n',
                '  .tbl caption{font:italic 16px "Trebuchet MS",Verdana,Arial,Helvetica,sans-serif;padding:0 0 5px;text-align:right;white-space:normal;}\n',
                '  .firstingroup {border-top:2px solid #6678B1;}\n',
                '</style>\n\n']

    html_out += ['<table class="tbl">\n',
                 '  <thead>\n',
                 '    <tr>\n',
                 '      <th align="center" valign="top"> Algorithm Name </th>\n',
                 '      <th align="center" valign="top"> Average Time </th>\n']
    for stat in stat_list:
        if stat not in stat_dict.keys():
            print("Error: unsupported statistic " + stat)
            sys.exit(1)
        html_out += ['      <th align="center" valign="top"> ' +
                             stat_dict[stat][0] +
                          ' </th>\n']
    html_out += ['    </tr>\n',
                 '  </thead>\n',
                 '  <tbody>\n']

    for algorithm in state.keys():
        arr = [state[algorithm][file]["angular_error"] for file in state[algorithm].keys() if file>=img_range[0] and file<=img_range[1]]
        average_time = "%.2f ms" % np.mean([state[algorithm][file]["time"] for file in state[algorithm].keys()
                                                                           if file>=img_range[0] and file<=img_range[1]])
        html_out += ['    <tr>\n',
                     '      <td>' + algorithm + '</td>\n',
                     '      <td>' + average_time + '</td>\n']
        for stat in stat_list:
            html_out += ['      <td> ' +
                                 "%.2f&deg" % stat_dict[stat][1](arr) +
                             ' </td>\n']
        html_out += ['    </tr>\n']
    html_out += ['  </tbody>\n',
                 '</table>\n']
    f = open(out, 'w')
    f.writelines(html_out)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("A benchmarking script for color balance algorithms"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-a",
        "--algorithms",
        metavar="ALGORITHMS",
        default="",
        help=("Comma-separated list of color balance algorithms to evaluate. "
              "Currently available: GT,learning_based,grayworld,nothing. "
              "Use a colon to set a specific model for the learning-based "
              "algorithm, e.g. learning_based:model1.yml,learning_based:model2.yml"))
    parser.add_argument(
        "-i",
        "--input_folder",
        metavar="INPUT_FOLDER",
        default="",
        help=("Folder containing input images to evaluate on. Assumes minimally "
              "processed png images like in the Gehler-Shi (http://www.cs.sfu.ca/~colour/data/shi_gehler/) "
              "or NUS 8-camera (http://www.comp.nus.edu.sg/~whitebal/illuminant/illuminant.html) datasets"))
    parser.add_argument(
        "-g",
        "--ground_truth",
        metavar="GROUND_TRUTH",
        default="real_illum_568..mat",
        help=("Path to the mat file containing ground truth illuminations. Currently "
              "supports formats supplied by the Gehler-Shi and NUS 8-camera datasets."))
    parser.add_argument(
        "-o",
        "--out",
        metavar="OUT",
        default="./white_balance_eval_result.html",
        help="Path to the output html table")
    parser.add_argument(
        "-s",
        "--state",
        metavar="STATE_JSON",
        default="./WB_evaluation_state.json",
        help=("Path to a json file that stores the current evaluation state"))
    parser.add_argument(
        "-t",
        "--stats",
        metavar="STATS",
        default="mean,median,p05,p20,p80,p95",
        help=("Comma-separated list of error statistics to compute and list "
              "in the output table. All the available ones are used by default"))
    parser.add_argument(
        "-b",
        "--input_bit_depth",
        metavar="INPUT_BIT_DEPTH",
        default="",
        help=("Assumed bit depth for input images. Should be specified in order to "
              "use full bit depth for evaluation (for instance, -b 12 for 12 bit images). "
              "Otherwise, input images are converted to 8 bit prior to the evaluation."))
    parser.add_argument(
        "-d",
        "--dst_folder",
        metavar="DST_FOLDER",
        default="",
        help=("If specified, this folder will be used to store the color correction results"))
    parser.add_argument(
        "-r",
        "--range",
        metavar="RANGE",
        default="0,0",
        help=("Comma-separated range of images from the dataset to evaluate on (for instance: 0,568). "
              "All available images are used by default."))
    parser.add_argument(
        "-m",
        "--model_folder",
        metavar="MODEL_FOLDER",
        default="",
        help=("Path to the folder containing models for the learning-based color balance algorithm (optional)"))
    args, other_args = parser.parse_known_args()

    if not os.path.exists(args.input_folder):
        print("Error: " + args.input_folder + (" does not exist. Please, correctly "
                                                 "specify the -i parameter"))
        sys.exit(1)

    if not os.path.exists(args.ground_truth):
        print("Error: " + args.ground_truth + (" does not exist. Please, correctly "
                                                 "specify the -g parameter"))
        sys.exit(1)

    state = {}
    if os.path.isfile(args.state):
        state = load_json(args.state)

    algorithm_list = parse_sequence(args.algorithms)
    img_range = list(map(int, parse_sequence(args.range)))
    if len(img_range)!=2:
        print("Error: Please specify the -r parameter in form <first_image_index>,<last_image_index>")
        sys.exit(1)

    img_files = sorted(os.listdir(args.input_folder))
    (gt_illuminants,black_levels) = load_ground_truth(args.ground_truth)

    for algorithm in algorithm_list:
        i = 0
        if algorithm not in state.keys():
            state[algorithm] = {}
        sz = len(img_files)
        for file in img_files:
            if file not in state[algorithm].keys() and\
             ((i>=img_range[0] and i<img_range[1]) or img_range[0]==img_range[1]==0):
                cur_path = os.path.join(args.input_folder, file)
                im = cv.imread(cur_path, -1).astype(np.float32)
                im -= black_levels[i]
                range_thresh = 255
                if len(args.input_bit_depth)>0:
                    range_thresh = 2**int(args.input_bit_depth) - 1
                    im = np.clip(im, 0, range_thresh).astype(np.uint16)
                else:
                    im = stretch_to_8bit(im)

                (time,angular_err) = evaluate(im, algorithm, gt_illuminants[i], i, range_thresh,
                                              256 if range_thresh > 255 else 64, args.dst_folder, args.model_folder)
                state[algorithm][file] = {"angular_error": angular_err, "time": time}
                sys.stdout.write("Algorithm: %-20s Done: [%3d/%3d]\r" % (algorithm, i, sz)),
                sys.stdout.flush()
                save_json(state, args.state)
            i+=1
    save_json(state, args.state)
    build_html_table(args.out, state, parse_sequence(args.stats), [img_files[img_range[0]], img_files[img_range[1]-1]])
