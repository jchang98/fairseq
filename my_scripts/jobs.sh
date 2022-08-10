base_dir=/SISDC_GPFS/Home_SE/xyduan-suda/cjin/fairseq
save=/SISDC_GPFS/Home_SE/xyduan-suda/cjin/fairseq_save
task_name=ldc-zhen-talkingheads
bsub -gpu num=1:mode=exclusive_process \
	-q HPC.S1.GPU.X785.suda \
	-J $task_name \
	-o $save/$task_name/checkpoints/jobs.out \
	-e $save/$task_name/checkpoints/jobs.err < $base_dir/my_scripts/train.sh