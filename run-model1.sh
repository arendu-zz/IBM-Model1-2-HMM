#!/bin/sh
SOURCE="data/dev.en"
TARGET="data/dev.es"
DEV_SOURCE="data/dev.en"
DEV_TARGET="data/dev.es"
# makes initial translation probabilities and saves them in data/inititial.trans
# -m uniform assigns uniform probabilities for every co-occuring word pair form source-traget
python initial_translation.py  -s $SOURCE -t $TARGET  -o data/initial.trans -m uniform

# report alignment error
# provided with assignment
# the f1 score should be 0.421 with these settings
echo "*** Model 1 Evaluation ***"
python eval_alignment.py data/dev.key data/model1.alignments

