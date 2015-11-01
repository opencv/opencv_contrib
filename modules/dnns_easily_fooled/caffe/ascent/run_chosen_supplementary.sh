#! /bin/bash -x

thisscript=$(readlink -f $0)
scriptdir=`dirname $thisscript`


for hp_seed in -1 169 188 360; do
    #for push_idx in 278 543 251 99 906 805; do
    for push_idx in 200 207 215 279 366 367 390 414 445 500 509 580 643 657 704 713 782 805 826 906; do
        for start_seed in `seq 0 4`; do
            startat=0

            seed_dir=`printf "seed_%04d" $hp_seed`
            result_dir="$scriptdir/results/supplementary_imgs/$seed_dir"
            mkdir -p $result_dir
            run_str=`printf 's%04d_idx%03d_sa%d_ss%02d' $hp_seed $push_idx $startat $start_seed`
            jobname="job_${run_str}"

            script="$result_dir/run_${run_str}.sh"
            result_prefix="$result_dir/$run_str"

            echo "#! /bin/bash" > $script
            echo "cd $scriptdir" >> $script
            echo "./hyperparam_search.py --result_prefix $result_prefix --hp_seed $hp_seed --push_idx $push_idx --start_seed $start_seed --startat $startat 2>&1" >> $script
            chmod +x $script

            qsub -N "$jobname" -A ACCOUNT_NAME -l nodes=1:ppn=2 -l walltime="1:00:00" -d "$result_dir" $script
        done
    done
done

