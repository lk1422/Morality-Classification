#####################################################################
#                   Convert Datasets into CSV                       # 
#####################################################################
import json
import csv

def remove_quotes(words):
    ret_string = ""
    for char in words:
        if char != '"':
            ret_string += char
    return ret_string

def parse_json(file1, file2):

    f = open(file1,'r')
    write_file = open(file2,'w')
    info = f.readlines()

    moral = True
    for line in info:
        objs = json.loads(line)
        out_text = objs["label"] + ',"' + remove_quotes(objs["situation"]) + " "
        if(moral):
            out_text +=  remove_quotes(objs["moral_action"]) + '"\n'
        else:
            out_text +=  remove_quotes(objs["immoral_action"]) + '"\n'
        write_file.write(out_text)

        moral = not moral
    f.close()
    write_file.close()

def parse_csv(file1, file2, flipped=True):
    #Flipped is true when the labels are flipped (1 should be moral)
    f = open(file1,'r')
    write_file = open(file2, 'w')
    reader =csv.reader(f, quotechar='"', delimiter=',', skipinitialspace=True)
    for row in reader:
        label = row[0]
        if flipped:
            label = str(abs(int(row[0])-1))
        write_file.write(label + ',"' + remove_quotes(row[1]) + '"\n')

    write_file.close()
    f.close()

def parse_moral_stories():
    parse_json("moral_stories/sec1/train.jsonl", "data/train.csv")
    parse_json("moral_stories/sec1/test.jsonl", "data/test.csv")
    parse_json("moral_stories/sec2/train.jsonl", "data/train2.csv")
    parse_json("moral_stories/sec2/test.jsonl", "data/test2.csv")
    parse_json("moral_stories/sec3/train.jsonl", "data/train3.csv")
    parse_json("moral_stories/sec3/test.jsonl", "data/test3.csv")

def parse_common_sense():
    parse_csv("ethics/commonsense/cm_train.csv", "data/cm_train.csv")
    parse_csv("ethics/commonsense/cm_test.csv", "data/cm_test.csv")

def parse_justice():
    parse_csv("ethics/justice/justice_train.csv", "data/justice_train.csv", False)
    parse_csv("ethics/justice/justice_test.csv", "data/justice_test.csv", False)

if __name__ == "__main__":
    parse_moral_stories()
    #parse_common_sense()
    #parse_justice()


    




