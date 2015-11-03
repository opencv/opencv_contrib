#! /usr/bin/env python

import argparse
import pickle
import pylab
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from collections import OrderedDict
import ipdb as pdb
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is normally in {caffe_root}/ascent. If it's elsewhere, change this path. 
import sys
sys.path.insert(0, caffe_root + 'python')
# If this next line fails, check the relevant paths.
import caffe

from misc_helper import *



def load_net_mean():
    # Pick which model to load, which image, etc.

    model_def_file = 'deploy_1_forcebackward.prototxt'

    # Can be downloaded from http://yosinski.cs.cornell.edu/yos_140311__caffenet_iter_450000
    pretrained_model = 'yos_140311__caffenet_iter_450000'

    # Can be downloaded from http://s.yosinski.com/synset_words.txt
    with open('%s/data/ilsvrc12/synset_words.txt' % caffe_root) as ff:
        labels = [line.strip() for line in ff.readlines()]

    # Load mean
    inmean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')

    offset = (256-227)/2
    mn = inmean[:, offset:offset+227, offset:offset+227]
    mni = mn.transpose((1,2,0))
    mnirgb = mni[:,:,::-1]           # convert to rgb order
    mn4d = mn[newaxis]

    net = caffe.Classifier(model_def_file, pretrained_model,
                           #mean=inmean,
                           channel_swap=(2,1,0),
                           #raw_scale=255.0,
                           #image_dims=(256, 256),
                           )

    net.set_phase_test()
    net.set_mode_cpu()

    return net, mnirgb, mn4d, labels



def update_result(result, suffix, ii, X, X0):
    result['iter_'+suffix] = ii
    result['norm_'+suffix] = norm(X)
    result['dist_'+suffix] = norm(X-X0)
    result['std_'+suffix] = X.flatten().std()
    result['X_'+suffix] = X.copy()



def find_image(net, mnirgb, mn4d, labels, decay = .01, N = 300, rseed = 0,
               push_layer = 'prob', push_idx = 278, start_at = 'mean_plus', prefix = 'junk',
               lr_policy = 'progress',
               lr_params = {'max_lr': 1e12, 'early_prog': .03, 'late_prog_mult': .1},
               blur_radius = 0,   # 0 or at least .3
               blur_every = 1,
               small_val_percentile = 0,
               small_norm_percentile = 0,
               px_benefit_percentile = 0,
               px_abs_benefit_percentile = 0):
    '''Find image for the given net using the specified start position, learning policies, etc.'''
    
    np.random.seed(rseed)

    #start_im = mnirgb[:] * 0
    if start_at == 'mean_plus':
        start_im = np.random.normal(0, 1, mnirgb.shape)
    elif start_at == 'randu':
        start_im = uniform(0, 255, mnirgb.shape) - mnirgb
    elif start_at == 'zero':
        start_im = zeros(mnirgb.shape)
    else:
        raise Exception('Unknown start conditions: %s' % start_at)

    if lr_policy == 'progress':
        assert 'max_lr' in lr_params
        assert 'early_prog' in lr_params
        assert 'late_prog_mult' in lr_params
    elif lr_policy == 'constant':
        assert 'lr' in lr_params
    else:
        raise Exception('Unknown lr_policy: %s' % lr_policy)

    try:
        push_idx = tuple(push_idx)   # tuple or list given
    except TypeError:
        push_idx = (push_idx, 0, 0)   # int given
    assert len(push_idx) == 3, 'provide push_idx in the form: int or (channel, x, y) tuple'
    
    #X0 = mn[newaxis,:]
    #im255 = im01 * 255 - 

    tmp = net.preprocess('data', start_im)   # converts rgb -> bgr
    X0 = tmp[newaxis,:]

    # What to change
    #push_idx = 278   # kit fox
    push_dir = 1.0
    class_unit = push_layer in ('fc8', 'prob')    # Whether or not the unit being optimized corresponds to one of the 1000 classes
    push_label = labels[push_idx[0]] if class_unit else 'None'
    
    X = X0.copy()
    #figsize(20,8)
    result = dict(
        iter_maj  = -1,
        iter_99   = -1,
        iter_999  = -1,
        iter_9999 = -1,
        iter_best = -1,
        norm_maj  = -1,
        norm_99   = -1,
        norm_999  = -1,
        norm_9999 = -1,
        norm_best = -1,
        dist_maj  = -1,
        dist_99   = -1,
        dist_999  = -1,
        dist_9999 = -1,
        dist_best = -1,
        std_maj   = -1,
        std_99    = -1,
        std_999   = -1,
        std_9999  = -1,
        std_best  = -1,
        act_best  = -1,
        X_maj     = None,
        X_99      = None,
        X_999     = None,
        X_9999    = None,
        X_best    = None,
        decay = decay, N = N, push_idx = push_idx, push_dir = push_dir, push_layer = push_layer,
        push_label = push_label,
        lr_policy = lr_policy, lr_params = lr_params,
        blur_radius = blur_radius, blur_every = blur_every,
        small_val_percentile = small_val_percentile, small_norm_percentile = small_norm_percentile,
        px_benefit_percentile = px_benefit_percentile, px_abs_benefit_percentile = px_abs_benefit_percentile,
    )

    print '\nParameters:'
    for key in sorted(result.keys()):
        print '%25s: %s' % (key, result[key])
    print
    
    for ii in range(N):
        X = minimum(255.0, maximum(0.0, X + mn4d)) - mn4d     # Crop all values to [0,255]
        out = net.forward_all(data = X)

        acts = net.blobs[push_layer].data
        
        iimax = unravel_index(acts.argmax(), acts.shape)[1:]   # chop off batch idx of 0
        obj = acts[0][push_idx]
        if ii > 0 and lr_policy == 'progress':
            print '   pred_prog: ', pred_prog, 'actual:', obj - old_obj
        if class_unit:
            print '%-4d' % ii, 'Push idx: %d, val: %g (%s)\n      Max idx: %d, val: %g (%s)' % (push_idx[0], acts[0][push_idx], push_label, iimax[0], acts.max(), labels[iimax[0]])
        else:
            print '%-4d' % ii, 'Push idx: %s, val: %g\n      Max idx: %s, val: %g' % (push_idx, acts[0][push_idx], iimax, acts.max())
        print '         X: ', X.min(), X.max(), norm(X)

        if acts[0][push_idx] > result['act_best']:
            update_result(result, 'best', ii, X, X0)
            result['acts_best'] = acts[0][push_idx]
        if iimax == push_idx and result['iter_maj'] == -1:
            update_result(result, 'maj', ii, X, X0)
        if acts[0][push_idx] > .99 and result['iter_99'] == -1:
            update_result(result, '99', ii, X, X0)
        if acts[0][push_idx] > .999 and result['iter_999'] == -1:
            update_result(result, '999', ii, X, X0)
        if acts[0][push_idx] > .9999 and result['iter_9999'] == -1:
            update_result(result, '9999', ii, X, X0)
            #break      # Quit once confidence > .9999

        diffs = net.blobs[push_layer].diff * 0
        diffs[0][push_idx] = push_dir
        backout = net.backward_from_layer(push_layer, diffs)

        grad = backout['data'].copy()
        print '      grad:', grad.min(), grad.max(), norm(grad)
        if norm(grad) == 0:
            print 'Grad 0, failed'
            break

        # progress-based lr
        if lr_policy == 'progress':
            late_prog = lr_params['late_prog_mult'] * (1-obj)
            desired_prog = min(lr_params['early_prog'], late_prog)
            prog_lr = desired_prog / norm(grad)**2
            lr = min(lr_params['max_lr'], prog_lr)
            print '    desired_prog:', desired_prog, 'prog_lr:', prog_lr, 'lr:', lr
            pred_prog = lr * dot(grad.flatten(), grad.flatten())
        elif lr_policy == 'constant':
            lr = lr_params['lr']
        else:
            raise Exception('Unimlemented lr_policy')

        print '     change size:', abs(lr * grad).max()
        old_obj = obj


        if ii < N-1:
            X += lr * grad
            X *= (1 - decay)

            if blur_radius > 0:
                if blur_radius < .3:
                    raise Exception('blur-radius of .3 or less works very poorly')
                oldX = X.copy()
                if ii % blur_every == 0:
                    for channel in range(3):
                        cimg = gaussian_filter(X[0,channel], blur_radius)
                        X[0,channel] = cimg
            if small_val_percentile > 0:
                small_entries = (abs(X) < percentile(abs(X), small_val_percentile))
                X = X - X*small_entries   # set smallest 50% of X to zero

            if small_norm_percentile > 0:
                pxnorms = norm(X, axis=1)
                smallpx = pxnorms < percentile(pxnorms, small_norm_percentile)
                smallpx3 = tile(smallpx[:,newaxis,:,:], (1,3,1,1))
                X = X - X*smallpx3
            
            if px_benefit_percentile > 0:
                pred_0_benefit = grad * -X
                px_benefit = pred_0_benefit.sum(1)
                smallben = px_benefit < percentile(px_benefit, px_benefit_percentile)
                smallben3 = tile(smallben[:,newaxis,:,:], (1,3,1,1))
                X = X - X*smallben3

            if px_abs_benefit_percentile > 0:
                pred_0_benefit = grad * -X
                px_benefit = pred_0_benefit.sum(1)
                smallaben = abs(px_benefit) < percentile(abs(px_benefit), px_abs_benefit_percentile)
                smallaben3 = tile(smallaben[:,newaxis,:,:], (1,3,1,1))
                X = X - X*smallaben3


    if class_unit:
        if result['iter_maj'] != -1:
            print 'Result: majority success'
        else:
            print 'Result: no convergence'

    for suffix in ('maj', '99', '999', '9999', 'best'):
        if result['X_'+suffix] is not None:
            asimg = net.deprocess('data', result['X_'+suffix])
            if suffix == 'best':
                best_X = asimg.copy()
            saveimagescc('%s_%s_X.jpg' % (prefix, suffix), asimg, 0)
            saveimagesc('%s_%s_Xpm.jpg' % (prefix, suffix), asimg + mnirgb)
        del result['X_'+suffix]
    with open('%s_info.pkl' % prefix, 'w') as ff:
        pickle.dump(result, ff)
    with open('%s_info.txt' % prefix, 'w') as ff:
        for key in sorted(result.keys()):
            print >>ff, key, result[key]

    return best_X


def main():
    parser = argparse.ArgumentParser(description='Finds images that activate a network in various ways.')
    parser.add_argument('--lr', type = float, default = .01)
    parser.add_argument('--decay', type = float, default = .01)
    parser.add_argument('--N', type = int, default = 300)
    parser.add_argument('--rseed', type = int, default = 0)
    parser.add_argument('--push_idx', type = int, default = -1)
    parser.add_argument('--start_at', type = str, default = 'mean_plus')
    parser.add_argument('--prefix', type = str, default = '%(push_idx)03d')
    parser.add_argument('--multi_idx_start', type = int, default = -1)
    parser.add_argument('--multi_idx_end', type = int, default = -1)
    args = parser.parse_args()

    assert (args.push_idx == -1) != (args.multi_idx_start == -1 and args.multi_idx_end == -1), 'Use push_idx xor multi*'
    assert (args.multi_idx_start == -1) == (args.multi_idx_end == -1), 'Use all multi* or none'

    net, mnirgb, mn4d, labels =  load_net_mean()

    if args.push_idx != -1:
        range_start = args.push_idx
        range_end = args.push_idx + 1
    else:
        range_start = args.multi_idx_start
        range_end = args.multi_idx_end
    for push_idx in range(range_start, range_end):
        prefix_dict = vars(args)
        prefix_dict['push_idx'] = push_idx
        prefix_str = args.prefix % prefix_dict
        print '\n\nFinding image'
        print 'prefix_str', prefix_str
        find_image(net, mnirgb, mn4d, labels,
                   lr = args.lr, decay = args.decay, N = args.N, rseed = args.rseed,
                   push_idx = args.push_idx, start_at = args.start_at,
                   prefix = prefix_str)



if __name__ == '__main__':
    main()
