#! /bin/bash -x

thisscript=$(readlink -f $0)
scriptdir=`dirname $thisscript`

for hp_seed in `seq 101 399`; do
#for hp_seed in 0; do
    for push_idx in 278 543 251 99 906 805; do
    #for push_idx in 278; do
        startat=0
        start_seed=0

        seed_dir=`printf "seed_%04d" $hp_seed`
        result_dir="$scriptdir/results/$seed_dir"
        mkdir -p $result_dir
        run_str=`printf 's%04d_idx%03d_sa%d_ss%02d' $hp_seed $push_idx $start_at $start_seed`
        jobname="job_${run_str}"

        script="$result_dir/run_${run_str}.sh"
        result_prefix="$result_dir/$run_str"

        echo "#! /bin/bash" > $script
        echo "cd $scriptdir" >> $script
        echo "./hyperparam_search.py --result_prefix $result_prefix --hp_seed $hp_seed --push_idx $push_idx --start_seed $start_seed --startat $startat 2>&1" >> $script
        chmod +x $script

        qsub -N "$jobname" -A ACCOUNT_NAME -l nodes=1:ppn=2 -l walltime="1:00:00" -d "$result_dir" $script

        #sleep 1
    done
done

