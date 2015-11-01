### General

This directory contains the necessary code to reproduce the gradient
ascent images in the paper: Figures 13, S3, S4, S5, S6, and S7. This
is research code, and so it may contain paths and such that are
particular to our setup that will need to be changed for your own
setup.

**Important note: this code requires the slightly modified version of caffe in this repository's [ascent](https://github.com/Evolving-AI-Lab/fooling/tree/ascent) branch. If you try running on master, you'll get an error about `backward_from_layer`.** See the below steps for using the correct branch.

If you find any bugs, please submit a PR!

If you have any trouble getting the code to work, please get in touch, and we'll help where we can.



### Notes on running the gradient ascent code

 * The gist of the gradient ascent code (along with a lot of
experimental bookkeeping) is in the
[find_image function in find_fooling_image.py](https://github.com/Evolving-AI-Lab/fooling/blob/master/caffe/ascent/find_fooling_image.py#L68-L274).
 * If you happen to be working in a
cluster environment that uses ```qsub```, you may find the shell scripts
useful; otherwise they probably won't help you much.
 * If you don't have a trained net around, you can download the trained model we used here: http://yosinski.cs.cornell.edu/yos_140311__caffenet_iter_450000
 * A file containing class labels is also used by the script and can be downloaded here: http://s.yosinski.com/synset_words.txt



### Simple steps to generate one fooling image

We'll walk through exact steps to generate a fooling image of a lion (class 291) using gradient ascent on the output unit for lion.

First, clone the repo and checkout the ascent branch:

    [~] $ git clone git@github.com:Evolving-AI-Lab/fooling.git
    [~] $ cd fooling
    [~/fooling] $ git checkout ascent
    [~/fooling] $ cd caffe

Configure and compile caffe. See [installation instructions](http://caffe.berkeleyvision.org/installation.html). Make sure to compile the python bindings too:

    [~/fooling/caffe] $ make -j && make -j pycaffe

Once Caffe is built, continue by fetching some auxiliary data (synsets.txt and a pre-trained model):

    [~/fooling/caffe] $ cd data/ilsvrc12
    [~/fooling/caffe/data/ilsvrc12] $ ./get_ilsvrc_aux.sh
    [~/fooling/caffe/data/ilsvrc12] $ cd ../../ascent
    [~/fooling/caffe/ascent] $ wget 'http://yosinski.cs.cornell.edu/yos_140311__caffenet_iter_450000'

Now we're ready to run the optimization. To find a quick fooling image for the Lion class (idx 291) using only 3 gradient steps, run the following:

    [~/fooling/caffe/ascent] $ ./find_fooling_image.py --push_idx 291 --N 3
    ...
    0    Push idx: 291, val: 0.00209935 (n02129165 lion, king of beasts, Panthera leo)
          Max idx: 815, val: 0.0114864 (n04275548 spider web, spider's web)
    ...
    1    Push idx: 291, val: 0.00962483 (n02129165 lion, king of beasts, Panthera leo)
          Max idx: 330, val: 0.0224016 (n02325366 wood rabbit, cottontail, cottontail rabbit)
    ...
    2    Push idx: 291, val: 0.0518007 (n02129165 lion, king of beasts, Panthera leo)
          Max idx: 291, val: 0.0518007 (n02129165 lion, king of beasts, Panthera leo)
    ...
    Result: majority success

