from flask import Flask
from flask import request
from flask import render_template

from vocab import Dictionary
from model import *
import utils

device = torch.device('cuda')
dictionary = Dictionary("words.dict")
model = torch.load("Transformer3.pth").to(device)
model.eval()

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def root():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        input_seq = request.form['input']
        input_tensor = utils.convert_input(input_seq, dictionary, 100)
        input_tensor = input_tensor.to(device)
        print(input_tensor)
        return render_template("output.html", moral=torch.sigmoid(model(input_tensor)) > 0.5)





