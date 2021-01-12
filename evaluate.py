# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:40:55 2020

@author: Zoe
"""
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist

def bleu(predict, target, n):
    return sentence_bleu([target], predict, weights=tuple(1 / n for i in range(n)))


def nist(predict, target, n):
    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)

def calculate_metrics(predict, reference):
    reference_len = len(reference)
    predict_len = len(predict)

    #-------------------bleu----------
    bleu_2 = bleu(predict, reference, 2)
    bleu_4 = bleu(predict, reference, 4)
    #-------------------nist----------
    nist_2 = nist(predict, reference, 2)
    nist_4 = nist(predict, reference, 4)
    #-------------------meteor----------
    predict = " ".join(predict)
    reference = " ".join(reference)
    meteor_scores = meteor_score([reference], predict)
    return bleu_2, bleu_4, nist_2, nist_4, meteor_scores

def evaluate_output(file_path):
    
    f_in = open(file_path, "r", encoding='utf-8')
    
    update_count = 0
    bleu_2scores = 0
    bleu_4scores = 0
    nist_2scores = 0
    nist_4scores = 0
    
    meteor_scores = 0
    while True:
        line = f_in.readline()
        if not line:
            break
        if line[:10] == "Reference:":
            reference = line[11:]
        elif line[:13] == "ground truth:":
            reference = line[14:]
            
        if line[:8] == "Predict:":
            predict = line[9:]

            temp_bleu_2, \
            temp_bleu_4, \
            temp_nist_2, \
            temp_nist_4, \
            temp_meteor_scores = calculate_metrics(predict[:-1], reference[:-1])

            bleu_2scores += temp_bleu_2
            bleu_4scores += temp_bleu_4
            nist_2scores += temp_nist_2
            nist_4scores += temp_nist_4

            meteor_scores += temp_meteor_scores
            update_count += 1
            
        elif line[:11] == "prediction:":
            predict = line[12:]

            temp_bleu_2, \
            temp_bleu_4, \
            temp_nist_2, \
            temp_nist_4, \
            temp_meteor_scores = calculate_metrics(predict[:-1], reference[:-1])

            bleu_2scores += temp_bleu_2
            bleu_4scores += temp_bleu_4
            nist_2scores += temp_nist_2
            nist_4scores += temp_nist_4

            meteor_scores += temp_meteor_scores
            update_count += 1
            
    print(f'test bleu_2scores: {bleu_2scores / update_count}')
    print(f'test bleu_4scores: {bleu_4scores / update_count}')
    print(f'test nist_2scores: {nist_2scores / update_count}')
    print(f'test nist_4scores: {nist_4scores / update_count}')
    print(f'test meteor_scores: {meteor_scores / update_count}')
    
    
if __name__ == '__main__':
    import os, optparse
    optparser = optparse.OptionParser()
    optparser.add_option("-o", "--outputfile", default='output/Bart_output.txt', help="output file created by the model")
    (opts, _) = optparser.parse_args()

    evaluate_output(opts.outputfile)

    
    
    