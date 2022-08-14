from imaplib import Commands
import os 
import random
import telebot
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

API_KEY = #Your API Key goes here 
bot = telebot.TeleBot(API_KEY)

@bot.message_handler()
def get_message(message1):
    message = tokenize(message1.text)
    X = bag_of_words(message, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, preds = torch.max(output, dim=1)
    tag = tags[preds.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][preds.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply_with = random.choice(intent['responses'])
                bot.reply_to(message1, reply_with)
    else:
        reply_with = "I don't understand..."
        bot.reply_to(message1, reply_with)

bot.polling() # This will keep checking for messages