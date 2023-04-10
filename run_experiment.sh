#!/bin/bash

# Define list of parameters
DOMAINS = ("finance" "reddit_eli5" "open_qa" "wiki_csai" "medicine")
SEQUENCE_LENGTHS = (32 128)

# Loop over training sequence lengths
for train_seq_len in "${SEQUENCE_LENGTHS[@]}"
do
    # Loop over the testing sequence lengths
    for test_seq_len in "${SEQUENCE_LENGTHS[@]}"
    do
        # Loop over training data domains
        for train_domain in "${DOMAINS[@]}"
        do
            # Loop over the testing data domains
            for test_domain in "${DOMAINS[@]}"
            do
                # Run the Python script with specified parameters
                python run.py --trainDomain $train_domain --testDomain $test_domain --trainSeqLength $train_seq_len --testSeqLength $test
            done
        done
    done
done
