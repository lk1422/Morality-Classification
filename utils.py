import string
import torch

def remove(word, removable):
    seq = ""
    for char in word:
        if char not in removable: 
            seq += char.lower()
    return seq
#********************************************************************#
#********************************************************************#
def replace(word, new, old):
    seq = ""
    for char in word:
        if char in old:
            seq+=new
        else:
            seq+=char
    return seq
#********************************************************************#
#********************************************************************#
def load_names(file_name):
    f = open(file_name, 'r')
    names = f.readlines()
    names = [name.strip().lower() for name in names]
    return set(names)
#********************************************************************#
#********************************************************************#
def convert_input(in_seq, dictionary, max_len):
    """Converts The input sequence into a input for the NN"""
    in_seq = in_seq.split()
    temp = dictionary.tokenize(in_seq)
    if len(temp) > max_len:
        temp = temp[:max_len-1]
        temp.append(dictionary.tokenize(["<EOS>"])[0])
    elif len(temp) < max_len:
        ext = dictionary.tokenize(["<PAD>"]) * (max_len - len(temp) - 1)
        temp.extend(ext)
        temp.append(dictionary.tokenize(["<EOS>"])[0])

    features = torch.LongTensor(temp).unsqueeze(0)
    return features
#********************************************************************#
#********************************************************************#
