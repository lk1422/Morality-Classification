from vocab import Dictionary
import csv
import torch

from torch.utils.data import Dataset

class EthicSet(Dataset):

    def __init__(self, input_file, dictionary, max_len):
        self.max_len = max_len
        self.dict = dictionary
        f = open(input_file, 'r')
        info = csv.reader(f, quotechar='"', delimiter=',', skipinitialspace=True)
        self.data = [row for row in info]
        f.close()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = int(self.data[i][0])
        temp = self.dict.tokenize(self.data[i][1].split())

        if len(temp) > self.max_len:
            temp = temp[:self.max_len-1]
            temp.append(self.dict.tokenize(["<EOS>"])[0])
        elif len(temp) < self.max_len:
            ext = self.dict.tokenize(["<PAD>"]) * (self.max_len - len(temp) - 1)
            temp.extend(ext)
            temp.append(self.dict.tokenize(["<EOS>"])[0])

        features = torch.LongTensor(temp)
        return(label, features)
        


if __name__ == "__main__":
    e = EthicSet("data/train_all.csv", "words.dict")
    for i in range(len(e)):
        print(e[i])

