KENLM=~/bin/kenlm/build/bin
TOK_INPUT=$1 #tokenized input trainset
MODEL_DIR=$2
TOK_INF=$3 #tokenized input testset
RES=$4

##training kenlm model
##output is the n_gram result in arpa format
time $KENLM/lmplz \
	-o 5 \
	--verbose_header \
	-S 80% \
	--text $TOK_INPUT \
	--arpa $MODEL_DIR/model_ngrams5_verbose.arpa \
	>& $MODEL_DIR/model_ngrams5_verbose.log
	
##compress the result to a binary file
time $KENLM/build_binary \
    -s $MODEL_DIR/model_ngrams5_verbose.arpa $MODEL_DIR/model_ngrams5_verbose.bin

##inference
python ./kenlm_inf.py $MODEL_DIR/model_ngrams5_verbose.bin $TOK_INF $RES
