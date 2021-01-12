import os
import pandas as pd
import json
import csv
from torch.utils.data import Dataset, DataLoader
import nlpaug.augmenter.word as naw


def create_en_dataset(txt_path, save_path, language='en'):

    patient = []
    doctor = []

    if language == 'en':
        with open(txt_path, 'r') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if line.startswith('Patient:'):
                    patient.append(' '.join(lines[i+1:i+2]))

                elif line.startswith('Doctor:'):
                    doctor.append(' '.join(lines[i+1: i+2]))

    data = {'src': patient, 'trg': doctor}
    df = pd.DataFrame.from_dict(data)
    df.to_csv(os.path.join(save_path, 'dialogue.csv'), index=False)


"""
Use this function if we are going to use Chinese dataset
"""
def read_chinese_json(src_json, language='cn'):

    df = pd.read_json(src_json)
    dialogue = df.Dialogue

    patient = []
    doctor = []

    for lines in dialogue:
        for line in lines:
            if line.startswith('病人'):
                patient.append(line)
            elif line.startswith('医生'):
                doctor.append(line)

    data = {'src': patient, 'trg': doctor}
    corpus = pd.DataFrame.from_dict(data)

    corpus.to_csv(os.path.join(
        save_path, 'full_{}.csv'.format(language)), index=False)

    train, val, test = split_by_fractions(corpus, [0.8, 0.1, 0.1])
    print('generating csv for train, validation and test')
    train.to_csv(os.path.join(
        save_path, 'train_{}.csv'.format(language)), index=False)
    val.to_csv(os.path.join(
        save_path, 'val_{}.csv'.format(language)), index=False)
    test.to_csv(os.path.join(
        save_path, 'test_{}.csv'.format(language)), index=False)


def removeprefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
        return string[:]


def split_by_fractions(df: pd.DataFrame, fracs: list, random_state: int = 42):
    assert sum(fracs) == 1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(
        sum(fracs))
    remain = df.index.copy().to_frame()
    res = []

    for i in range(len(fracs)):
        fractions_sum = sum(fracs[i:])
        frac = fracs[i]/fractions_sum
        idxs = remain.sample(frac=frac, random_state=random_state).index
        remain = remain.drop(idxs)
        res.append(idxs)
    return [df.loc[idxs] for idxs in res]


def save_df(train, test, validation, save_dir):

    train.to_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    validation.to_csv(os.path.join(save_dir, 'valid.tsv'), sep='\t')
    test.to_csv(os.path.join(save_dir, 'test.tsv'), sep='\t')


def augment_dataset(csv, model_dir):

    original = pd.read_csv(csv)

    """
    Conduct two process of augmentation
    1. Synonym augmentation
    2. Word Embedding augmemntation
    """

    syn_df = original.copy()
    syn_aug = naw.SynonymAug(aug_src='wordnet')

    # synonym augenter(simple version)
    for i, query in enumerate(syn_df.src):
        synonym = syn_aug.augment(query)
        syn_df.at[i, 'src'] = synonym

    #word embedding augmenter
    word_df = original.copy()
    embed_aug = naw.WordEmbsAug(
        model_type='fasttext', model_path=model_dir+'/wiki-news-300d-1M.vec',
        action="insert")

    for i, query in enumerate(word_df.src):
        insertion = embed_aug.augment(query)
        word_df.at[i, 'src'] = insertion

    a1 = pd.catcat([original, syn_df])
    a2 = pd.concat([a1, word_df])

    a2.to_csv(os.path.join(model_dir, 'augmented.csv'), index=False)

    return a2


if __name__ == "__main__":

    txt_path = 'data/covid_additional.txt'
    save_path = 'vanilla_attention/split_data'

#    create_en_dataset(txt_path, save_path)

    original = 'vanilla_attention/split_data/dialogue.csv'

    augment_dataset(original, save_path)
