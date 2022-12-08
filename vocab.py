###############################################################
#           Dictionary/Tokenizer for Words in Dataset         #
###############################################################

import csv
import utils
import string

class Dictionary:
    def __init__(self, dict_file, names=None):
        self.dict = {}
        self.tokens = []
        self.names = names
        self.read_dict(dict_file)
#********************************************************************#
#********************************************************************#
    def create_dict(train_file,dict_file, names=None):
        """
        Takes in a file and converts all the words to tokens
        MUST LOAD NAMES BEFORE RUNNING
        """
        #OPEN FILES
        f = open(train_file, 'r')
        w = open(dict_file, 'w')

        words_seen = set()
        total_words = 0

        reader =csv.reader(f, quotechar='"', delimiter=',', skipinitialspace=True)
        #REMOVE PUNCTUATION AND FIX CAPTIALIZATION BEFORE SPLIT
        for row in reader:
            words = utils.remove(row[1], string.punctuation)
            words = utils.replace(words, ' ', ['\n', '\t'])
            words = words.split()

        #WRITE UNIQUE WORDS TO DICT FILE
            for word in words:
                if (word not in words_seen and ( names == None or word not in names )) and word != "":
                    w.write(word + " " + str(total_words) + "\n")
                    words_seen.add(word)
                    total_words += 1

        #ADD UTILITY TOKENS
        w.write("<SOS> " + str(total_words)   + "\n")
        w.write("<EOS> " + str(total_words+1) + "\n")
        w.write("<PAD> " + str(total_words+2) + "\n")
        w.write("<UNK> " + str(total_words+3) + "\n")
        w.write("<NAM> " + str(total_words+4) + "\n")


        w.close()
        f.close()
#********************************************************************#
#********************************************************************#
    def read_dict(self, dict_file):
        f = open(dict_file, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip()
            info = line.split()
            self.dict[info[0]] = int(info[1])
            self.tokens.append(info[0])
#********************************************************************#
#********************************************************************#
    def tokenize(self, sequence):
        """
        Takes in a sequence of strings/words
        Outputs a list of tokens for the sequence
        MUST LOAD NAMES BEFORE RUNNING
        """
        sequence = utils.remove(' '.join(sequence), string.punctuation).split(' ')# FIX THIS LATER MAKE REMOVE ACCEPT LIST

        tokens = []
        for word in sequence:
            word = word.lower()
            if word in self.dict:
                tokens.append(self.dict[word])
            elif self.names != None and word in self.names:
                tokens.append(self.dict["<NAM>"])
            else:
                tokens.append(self.dict["<UNK>"])
        return tokens
#********************************************************************#
#********************************************************************#
    def stringify(self, tokens):
        """
        Takes in a sequence of tokens
        Outputs a list of words
        MUST LOAD NAMES BEFORE RUNNING
        """
        words = []
        for token in tokens:
            words.append(self.tokens[int(token)])
        return words
#********************************************************************#
#********************************************************************#
    def __len__(self):
        return len(self.dict)
#********************************************************************#
#********************************************************************#

if __name__ == "__main__":
    names = utils.load_names("data/names.txt")
    Dictionary.create_dict("data/vocab.csv", "words.dict")

    d = Dictionary("words.dict", names)
    test = "hello, my name is lenny! testing test testing blank".split()
    test2  = d.tokenize(test)
    print(test2)
    print(d.stringify(test2))
