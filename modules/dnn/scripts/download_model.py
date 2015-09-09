#!/usr/bin/env python
import os
import sys
import time
import urllib
import hashlib
import argparse
import json


def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    global prev_duration
    if count == 0:
        start_time = time.time()
        prev_duration = -1
        return
    duration = max(1, time.time() - start_time)
    if int(duration) == int(prev_duration):
        return

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
    prev_duration = duration


# Function for checking SHA1.
def model_checks_out(filename, sha1):
    with open(filename, 'r') as f:
        return hashlib.sha1(f.read()).hexdigest() == sha1

def model_download(filename, url, sha1):
    # Check if model exists.
    if os.path.exists(filename) and model_checks_out(filename, sha1):
        print("Model {} already exists.".format(filename))
        return

    # Download and verify model.
    urllib.urlretrieve(url, filename, reporthook)
    print model_checks_out(filename, sha1)
    if not model_checks_out(filename, sha1):
        print("ERROR: model {} did not download correctly!".format(url))
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downloading trained model binaries.")
    parser.add_argument("download_list")
    args = parser.parse_args()

    test_dir = os.environ.get("OPENCV_TEST_DATA_PATH")
    if not test_dir:
        print "ERROR: OPENCV_TEST_DATA_PATH environment not specified"
        sys.exit(1)

    try:
        with open(args.download_list, 'r') as f:
            models_to_download = json.load(f)
    except:
        print "ERROR: Can't pasrse {}".format(args.download_list)
        sys.exit(1)

    for model_name in models_to_download:
        model = models_to_download[model_name]

        dst_dir = os.path.join(test_dir, os.path.dirname(model['file']))
        dst_file = os.path.join(test_dir, model['file'])
        if not os.path.exists(dst_dir):
            print "ERROR: Can't find module testdata path '{}'".format(dst_dir)
            sys.exit(1)

        print "Downloading model '{}' to {} from {} ...".format(model_name, dst_file, model['url'])
        model_download(dst_file, model['url'], model['sha1'])