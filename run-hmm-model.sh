#!/bin/sh
SOURCE="data/dev.en"
TARGET="data/dev.es"
DEV_SOURCE="data/dev.en"
DEV_TARGET="data/dev.es"
# step 1
# makes initial translation probabilities and saves them in data/inititial.trans
# -m uniform assigns uniform probabilities for every co-occuring word pair form source-traget
python initial_translation.py  -s $SOURCE -t $TARGET  -o data/initial.trans -m uniform

# step 2
# run model1 alignment using initital translation probabilities
python model1.py -s $SOURCE -t $TARGET  -i data/initial.trans -p data/model1.trans -a data/model1.alignments  -as $DEV_SOURCE -at $DEV_TARGET

# step 3
# run model2 alignment using output from model1
#python model2.py -s $SOURCE -t $TARGET  -i data/model1.trans -p data/model2.trans -a data/model2.alignments  -as $DEV_SOURCE -at $DEV_TARGET

# step 4
# run hmm model alignment using output from model1
python hmm-jump-model.py  -s $SOURCE -t $TARGET -ia data/model1.alignments -it data/model1.trans -p data/hmm.trans -a data/hmm.alignments -as $DEV_SOURCE -at $DEV_TARGET

# step 5
# report alignment error
# provided with assignment
# the f1 score should be 0.421 with these settings
echo "*** Model 1 Evaluation ***"
python eval_alignment.py data/dev.key data/model1.alignments
# the f1 score should be 0.449 with these settings
#python eval_alignment.py data/dev.key data/model2.alignments
# the f1 score should be more than 0.449 with these settings
echo "*** HMM Model Evaluation ***"
python eval_alignment.py data/dev.key data/hmm.alignments-4
