#! /usr/bin/env python

from pylab import *
import os
import argparse
import ipdb as pdb

from find_fooling_image import load_net_mean, find_image



def rchoose(choices, prob=None):
    if prob is None:
        prob = ones(len(choices))
    prob = array(prob, dtype='float')
    return np.random.choice(choices, p=prob/prob.sum())



def main():
    parser = argparse.ArgumentParser(description='Hyperparam search')
    parser.add_argument('--result_prefix', type = str, default = './junk')
    parser.add_argument('--hp_seed', type = int, default = 0)
    parser.add_argument('--start_seed', type = int, default = 0)
    parser.add_argument('--push_idx', type = int, default = 278)
    parser.add_argument('--layer', type = str, default = 'prob', choices = ('fc8', 'prob'))
    parser.add_argument('--startat', type = int, default = 0, choices = (0, 1))
    args = parser.parse_args()

    push_idx = args.push_idx
    small_val_percentile = 0
    start_at = 'mean_plus' if args.startat == 0 else 'randu'    
    
    if args.hp_seed == -1:
        # Special hp_seed of -1 to do gradient descent without any regularization
        decay = 0
        N = 500
        early_prog = .02
        late_prog_mult = .1
        blur_radius = 0
        blur_every = 1
        small_norm_percentile = 0
        px_benefit_percentile = 0
        px_abs_benefit_percentile = 0
    else:
        np.random.seed(args.hp_seed)

        # Choose hyperparameter values given this seed
        decay = rchoose((0, .0001, .001, .01, .1, .2, .3),
                        (4,     1,    1,   2,  1,  1,  1))
        N = rchoose((250, 500, 750, 1000, 1500))
        early_prog = rchoose(
            (.02, .03, .04),
            (1, 2, 1))
        late_prog_mult = rchoose((.02, .05, .1, .2))
        blur_radius = rchoose(
            (0, .3, .4, .5, 1.0),
            (10, 2,  1,  1,  1))
        blur_every = rchoose((1, 2, 3, 4))
        small_norm_percentile = rchoose(
            (0, 10, 20, 30, 50, 80, 90),
            (10, 10, 5, 2,  2,  2,  2))
        px_benefit_percentile = rchoose(
            (0,  10, 20, 30, 50, 80, 90),
            (20, 10,  5,  2,  2,  2,  2))
        px_abs_benefit_percentile = rchoose(
            (0,  10, 20, 30, 50, 80, 90),
            (10, 10,  5,  2,  2,  2,  2))

    prefix = args.result_prefix
    print 'prefix is', prefix
    
    net, mnirgb, mn4d, labels =  load_net_mean()

    find_image(net, mnirgb, mn4d, labels,
               decay = decay,
               N = N,
               rseed = args.start_seed,
               push_idx = push_idx,
               start_at = start_at,
               prefix = prefix,
               lr_policy = 'progress',
               lr_params = {'max_lr': 1e7,
                            'early_prog': early_prog,
                            'late_prog_mult': late_prog_mult},
               blur_radius = blur_radius,
               blur_every = blur_every,
               small_val_percentile = small_val_percentile,
               small_norm_percentile = small_norm_percentile,
               px_benefit_percentile = px_benefit_percentile,
               px_abs_benefit_percentile = px_abs_benefit_percentile,
           )



if __name__ == '__main__':
    main()
