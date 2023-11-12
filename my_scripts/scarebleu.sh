### sacrebleu内部默认采用13a进行分词，对于中文和日语需要自己指定（zh,ja-mecab）
### fairseq 验证集中报的分数是sacrebleu的分数，采用的是13a进行分词。
### 需要传递一个参数eval_bleu_detok_args,然后更改fairseq/tasks/translation.py中的inference_with_bleu函数去传递tokenizer。

base_dir=xxx/fairseq/project/wmt22
task_name=wmt-enzh-big-baseline
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


checkpoint_path=$base_dir/checkpoints/$task_name/checkpoint_last.pt
prefix_dir=~/cjin/data
data_bin=/home/notebook/data/personal/S9048436/wmt/wmt22_en_zh/data-bin/


tmp=$base_dir/tmp/$task_name
result=$base_dir/bleu_score
mkdir -p $tmp

beam=5
buffer_size=1024
max_tokens=10000

CUDA_VISIBLE_DEVICES=$num fairseq-generate $data_bin \
    --gen-subset test \
    --path $checkpoint_path \
    --beam $beam \
    --batch-size 128 \
    --remove-bpe \
    | tee $tmp/generate.out

grep ^H $tmp/generate.out  \
| sed 's/^H\-//' \
| sort -n -k 1 \
| cut -f 3 \
| sacremoses detokenize \
> $tmp/generate.hyp.detok

ref=newtest2022.enzh.zh

sacrebleu $ref -tok zh  -m bleu < $tmp/generate.hyp.detok

echo "----------------------------------------------------------------" >> $result
echo "the sacrebleu of $task_name" >> $result
sacrebleu $ref -tok zh  -m bleu < $tmp/generate.hyp.detok >> $result
