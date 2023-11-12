base_dir=xxx/fairseq
save=xxx/fairseq_save
task_name=ldc-zhen-talkingheads
save_dir=$save/$task_name/checkpoints/
log_file=$save/$task_name/checkpoints/train.log
mkdir -p $save_dir

data_bin=~/cjin/data/ldc/corpus/data-bin/
# restore_file=$base_dir/checkpoints/wmt-en2de/checkpoint_best.pt

arch=varient_transformer
criterion=label_smoothed_cross_entropy
# criterion=reg_label_smoothed_cross_entropy
label_smoothing=0.1
dropout=0.3
lr=0.0005
lr_scheduler=inverse_sqrt
extra='--fp16'

config_k=(save_dir arch criterion label_smoothing dropout lr lr_scheduler extra)
config_v=($save_dir $arch $criterion $label_smoothing $dropout $lr $lr_scheduler $extra)

for ((i=0;i<${#config_k[@]};i++)); do 
    echo ${config_k[$i]}=${config_v[$i]}
done

num=0
max_tokens=409

export CUDA_VISIBLE_DEVICES=3
train(){
    fairseq-train $data_bin \
        --save-dir $save_dir \
        --optimizer adam \
        --stop-min-lr  1e-09 \
        --lr $lr --clip-norm 0.0 \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --lr-scheduler $lr_scheduler \
        --dropout $dropout \
        -s ch -t en \
        --arch $arch \
        --warmup-init-lr 1e-7 \
        --warmup-updates 4000 \
        --weight-decay 0.0 \
        --adam-betas '(0.9,0.98)' \
        --max-tokens $max_tokens \
        --log-interval 100 \
        --no-progress-bar \
        --patience 20  \
        --keep-last-epochs 10 \
        --share-decoder-input-output-embed \
        --eval-bleu \
        --eval-bleu-detok moses \
        --eval-bleu-detok-args '{"tokenize" : "zh"}' \
        --user-dir $base_dir/plugin \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --update-freq 2 \
        --encoder-talking-heads True \
        --decoder-talking-heads True \
        $extra | tee $log_file

}

train
        # --eval-bleu-args '{"beam": 5, "lenpen": 0.6}' \
        # --restore-file $restore_file --reset-lr-scheduler --reset-optimizer \
