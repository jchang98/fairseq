base_dir=/SISDC_GPFS/Home_SE/xyduan-suda/cjin/fairseq
save=/SISDC_GPFS/Home_SE/xyduan-suda/cjin/fairseq_save
task_name=ldc-zhen-baseline
avg_epoch=0

num=0
if test $avg_epoch -ne 0;then
    echo "averaged the checkpoints"
    python scripts/average_checkpoints.py \
        --inputs $base_dir/checkpoints/$task_name/ \
        --output $base_dir/checkpoints/$task_name/checkpoint_avg$avg_epoch.pt \
        --num-epoch-checkpoint $avg_epoch
else
    echo "not averaged the checkpoints"
fi


checkpoint_path=$save/$task_name/checkpoints/checkpoint_best.pt
prefix_dir=~/cjin/data
bleu=$prefix_dir/multi-bleu.perl
data_bin=~/cjin/data/ldc/corpus/data-bin/
nist=$prefix_dir/ldc/corpus/ldc/nist


tmp=$base_dir/tmp/$task_name
result=$base_dir/bleu_score
mkdir -p $tmp

beam=5
buffer_size=1024
max_tokens=10000

for i in 02 03 04 05 08;do
CUDA_VISIBLE_DEVICES=$num fairseq-interactive $data_bin \
    --path $checkpoint_path \
    --beam $beam \
    --buffer-size $buffer_size \
    --max-tokens $max_tokens \
    --remove-bpe \
    -s ch -t en \
    --input $nist/nist$i.bpe.in > $tmp/nist$i.$task_name.tmp

grep ^H $tmp/nist$i.$task_name.tmp | cut -f3- > $tmp/nist$i.$task_name

perl $bleu -lc $nist/nist$i.ref.* < $tmp/nist$i.$task_name > $tmp/nist$i.$task_name.score

done

write(){
    echo "----------------------------------------------------------------"
    echo "the bleu of $task_name"
    cat $tmp/nist0*.$task_name.score
    cat $tmp/nist0*.$task_name.score | grep "BLEU = ....." -o | awk '{sum+=$3 } END {print "AVG = ", sum/NR}'
}

write >> $result
cat $tmp/nist0*.$task_name.score
