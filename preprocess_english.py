import os
import pandas as pd
import json

from pytorch_pretrained_bert import BertTokenizer
import torch
import codecs

"""
- Simple function to rid data on txt file and split by ids
- You can read the txt file as dataframe by using the section function
"""

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_ENCODER_SIZE = 400
MAX_DECODER_SIZE = 100


def clean_dataset_1(dataset_file, json_file):
    '''
    save dataset_file in json file
    '''
    f_in = open(dataset_file, "r", encoding='utf-8')
    f_json = open(json_file, "w", encoding='utf-8')
    
    total = 0
    last_part = ""
    last_turn = 0
    last_dialog = {}
    last_list = []
    last_user = ""
    
    Dialog_list = []
    
    check_list = []
    
    while True:
        line = f_in.readline()
        if not line:
            break
        if line[:11] == "Description":
            last_part = "description"
            last_turn = 0
            last_dialog = {}
            last_list = []
            last_user = ""
            last_utterance = ""
            while True:
                line = f_in.readline()
                if (not line) or (line in ["\n", "\n\r"]):
                    break
                #if line[:5] == '病情描述：':
                last_user = "Patient:"
                sen = line.rstrip()
                if sen == "":
                    continue
                if sen[-1] not in '.。？，,?!！~～':
                    sen += '。'
                if sen in check_list:
                    last_utterance = ""
                # else:
                #     last_utterance = last_user# + sen
                #     check_list.append(sen)
                break
            
        elif line[:8] == "Dialogue":
            if last_part == "description":
                if len(last_utterance) >= 0:
                    last_part = "dialogue"
                    last_user = "Patient:"
                    last_turn = 1
                    while True:
                        line = f_in.readline()
                        if (not line) or (line in ["\n", "\n\r"]):
                            last_user = ""
                            last_list.append(last_utterance)
    
                            if  int(last_turn / 2) > 0:
                                temp = int(last_turn / 2)
                                last_dialog["Turn"] = temp
                                total += 1
                                last_dialog["Id"] = total
                                last_dialog["Dialogue"] = last_list[: temp * 2]
                                Dialog_list.append(last_dialog)
                            break
                            
                        if line[:8] == "Patient:" or line[:7] == "Doctor:":
                            user = 'Patient:' if line[:8]=='Patient:' else 'Doctor:'
                            line = f_in.readline()
                            sen = line.rstrip()
                            # if sen == "":
                            #     continue
    
                            if sen[-1] not in '.。？，,?!！~～':
                                sen += '.'
                                
                            if user == last_user:
                                last_utterance = last_user + last_utterance + sen
                            else:
                                last_user = user
                                last_list.append(last_utterance)
                                last_turn += 1
                                last_utterance = user + sen
                        else:
                            sen = line.rstrip()
                            if user == last_user:
                                last_utterance = last_utterance + sen
                                

    print ("Total Cases: ", total)
    json.dump(Dialog_list, f_json, ensure_ascii = False, indent = 4)
    f_in.close()
    f_json.close()


def get_splited_data_by_file(dataset_file):
    datasets = [[], [], []]

    with open(dataset_file, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)

    total_id_num = len(data)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    datasets[0] = [d['Dialogue'] for d in data[:validate_idx]]
    datasets[1] = [d['Dialogue'] for d in data[validate_idx:test_idx]]
    datasets[2] = [d['Dialogue'] for d in data[test_idx:]]
    
    return datasets


def seq2token_ids(source_seqs, target_seq):
    # 可以尝试对source_seq进行切分
    encoder_input = []
    for source_seq in source_seqs:
        # remove xx：
        if source_seq[:8] == 'Patient:':
            encoder_input += tokenizer.tokenize(source_seq[8:]) + ["[SEP]"]
        else:
            encoder_input += tokenizer.tokenize(source_seq[7:]) + ["[SEP]"]

    decoder_input = ["[CLS]"] + tokenizer.tokenize(target_seq[7:])  # remove Doctor：

    # 设置不得超过 MAX_ENCODER_SIZE 大小
    if len(encoder_input) > MAX_ENCODER_SIZE - 1:
        if "[SEP]" in encoder_input[-MAX_ENCODER_SIZE:-1]:
            idx = encoder_input[:-1].index("[SEP]", -(MAX_ENCODER_SIZE - 1))
            encoder_input = encoder_input[idx + 1:]

    encoder_input = ["[CLS]"] + encoder_input[-(MAX_ENCODER_SIZE - 1):]
    decoder_input = decoder_input[:MAX_DECODER_SIZE - 1] + ["[SEP]"]
    # enc_len = len(encoder_input)
    # dec_len = len(decoder_input)
    
    # conver to ids
    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)

    # mask
    mask_encoder_input = [1] * len(encoder_input)
    mask_decoder_input = [1] * len(decoder_input)

    # padding
    encoder_input += [0] * (MAX_ENCODER_SIZE - len(encoder_input))
    decoder_input += [0] * (MAX_DECODER_SIZE - len(decoder_input))
    mask_encoder_input += [0] * (MAX_ENCODER_SIZE - len(mask_encoder_input))
    mask_decoder_input += [0] * (MAX_DECODER_SIZE - len(mask_decoder_input))

    # turn into tensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    
    mask_encoder_input = torch.LongTensor(mask_encoder_input)
    mask_decoder_input = torch.LongTensor(mask_decoder_input)

    return encoder_input, decoder_input, mask_encoder_input, mask_decoder_input


def make_dataset(data, file_name='train_data.pth'):
    train_data = []

    for d in data:
        d_len = len(d)
        for i in range(d_len // 2):
            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[:2 * i + 1], d[2 * i + 1])
            train_data.append((encoder_input, 
                            decoder_input,
                            mask_encoder_input,
                            mask_decoder_input))

    encoder_input, \
    decoder_input, \
    mask_encoder_input, \
    mask_decoder_input = zip(*train_data)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    mask_encoder_input = torch.stack(mask_encoder_input)
    mask_decoder_input = torch.stack(mask_decoder_input)

    # if file_name == 'test_data.pth':
    #     train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, input_text, reference_text]
    # else:
    train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input]

    torch.save(train_data, file_name)
    
    
def make_test_dataset(data, file_name='test_data.pth'):
    train_data = []
    text_data = []

    for d in data:
        d_len = len(d)
        for i in range(d_len // 2):
            #print(d[2*i])
            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[:2 * i+1], d[2 * i + 1])
            train_data.append((encoder_input, 
                            decoder_input,
                            mask_encoder_input,
                            mask_decoder_input))
            text_data.append((d[2 * i],
                            d[2 * i + 1]))

    encoder_input, \
    decoder_input, \
    mask_encoder_input, \
    mask_decoder_input= zip(*train_data)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    mask_encoder_input = torch.stack(mask_encoder_input)
    mask_decoder_input = torch.stack(mask_decoder_input)

    # if file_name == 'test_data.pth':
    #     train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input, input_text, reference_text]
    # else:
    train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input]

    torch.save(train_data, file_name)
    
    with open('text_data.json', 'w') as f:
        json.dump(text_data, f)
    
if __name__ == "__main__":
    # text_path = 'data/raw.txt'
    sav_dir = 'data'

    # raw_txt = read_raw_txt(text_path)
    # json_file = os.path.join(sav_dir, 'cleaned.json')
    
    dataset_file = os.path.join(sav_dir, 'covid_additional.txt')
    json_file = os.path.join(sav_dir, 'cleaned.json')
    
    clean_dataset_1(dataset_file, json_file)
    data = get_splited_data_by_file(json_file)
    
    print(f'Process the train dataset')
    make_dataset(data[0], 'train_data.pth')
    
    print(f'Process the validate dataset')
    make_dataset(data[1], 'validate_data.pth')
    
    print(f'Process the test dataset')
    make_dataset(data[2], 'test_data.pth')
  


