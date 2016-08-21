import argparse
import glob
import os
import subprocess


def execute(cmd):
    popen = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    for stdout_line in iter(popen.stdout.readline, ''):
        print(stdout_line.rstrip())
    for stderr_line in iter(popen.stderr.readline, ''):
        print(stderr_line.rstrip())
    popen.stdout.close()
    popen.stderr.close()
    return_code = popen.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Train Global Patch Collider using Middlebury dataset')
    parser.add_argument(
        '--bin_path',
        help='Path to the training executable (example_optflow_gpc_train)',
        required=True)
    parser.add_argument('--dataset_path',
                        help='Path to the directory with frames',
                        required=True)
    parser.add_argument('--gt_path',
                        help='Path to the directory with ground truth flow',
                        required=True)
    parser.add_argument('--descriptor_type',
                        help='Descriptor type',
                        type=int,
                        default=0)
    args = parser.parse_args()
    seq = glob.glob(os.path.join(args.dataset_path, '*'))
    seq.sort()
    input_files = []
    for s in seq:
        if os.path.isdir(s):
            seq_name = os.path.basename(s)
            frames = glob.glob(os.path.join(s, 'frame*.png'))
            frames.sort()
            assert (len(frames) == 2)
            assert (os.path.basename(frames[0]) == 'frame10.png')
            assert (os.path.basename(frames[1]) == 'frame11.png')
            gt_flow = os.path.join(args.gt_path, seq_name, 'flow10.flo')
            if os.path.isfile(gt_flow):
                input_files += [frames[0], frames[1], gt_flow]
    execute([args.bin_path, '--descriptor-type=%d' % args.descriptor_type] + input_files)


if __name__ == '__main__':
    main()
