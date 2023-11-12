base_dir=xxx/fairseq
save=xxx/fairseq_save/layerkd
task_name=iwslt-deen-pkd
avg_epoch=0

num=1
if test $avg_epoch -ne 0;then
    echo "averaged the checkpoints"
    python $base_dir/scripts/average_checkpoints.py \
        --inputs $save/$task_name/checkpoints/ \
        --output $save/$task_name/checkpoints/checkpoint_avg$avg_epoch.pt \
        --num-epoch-checkpoint $avg_epoch
else
    echo "not averaged the checkpoints"
fi


checkpoint_path=$save/$task_name/checkpoints/checkpoint_avg5.pt
prefix_dir=~/cjin/data
bleu=$prefix_dir/multi-bleu.perl
# data_bin=~/cjin/data/ldc/corpus/data-bin/
data_bin=~/cjin/data/iwslt14/iwslt-bt-pkd/data-bin/
detok=~/cjin/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl
tok=~/cjin/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl
# nist=$prefix_dir/ldc/corpus/nist


tmp=$save/$task_name/gen_out
result=$save/bleu_score
mkdir -p $tmp

beam=5
buffer_size=1024
max_tokens=20000

gen_sub=valid
CUDA_VISIBLE_DEVICES=$num fairseq-generate $data_bin \
    --gen-subset $gen_sub \
    --path $checkpoint_path \
    --beam $beam \
    --max-tokens $max_tokens \
    --remove-bpe \
    -s de -t en \
    --user-dir $base_dir/plugin \
    | tee  $tmp/$gen_sub.out

grep ^H $tmp/$gen_sub.out | cut -f3-  > $tmp/$gen_sub.sys
grep ^T $tmp/$gen_sub.out | cut -f2-  > $tmp/$gen_sub.ref

perl $bleu -lc $tmp/$gen_sub.ref < $tmp/$gen_sub.sys 


