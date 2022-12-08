from ethicset import EthicSet
from vocab import Dictionary
from model import Transformer2



v = Dictionary("words.dict")
e = EthicSet("data/train_all.csv", v, 200)
for x,y in e:
    print(x,y)
    print()


