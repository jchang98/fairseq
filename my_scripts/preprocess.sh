data_dir=$1
dir=$data_dir/data-bin

preprocess(){

fairseq-preprocess \
        --trainpref $data_dir/train.bpe \
        --validpref $data_dir/valid.bpe \
        --testpref $data_dir/test.bpe \
        --source-lang de --target-lang en \
        --destdir  $dir \
        --workers 20  \
        --srcdict $data_dir/dict.de.txt --tgtdict $data_dir/dict.en.txt 
        

}

preprocess