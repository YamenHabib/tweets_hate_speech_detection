import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
from transformers import RobertaTokenizer, RobertaModel

from model import *
parser = argparse.ArgumentParser(description='Evaluate the hate speach of a tweet.')
parser.add_argument('tweet', action='store', type=str, help='The text of the tweet to evaluate')

args = parser.parse_args()
tweet = args.tweet

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device('cpu')


def load_checkpoint(path, model):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']
  
model = ROBERTAClassifier(0.4)
path = "model.pkl"
load_checkpoint(path, model)

model = model.to(device)
model.eval()

# Initialize tokenizer.
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

print(tweet)
tokens = tokenizer.tokenize(tweet)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
source = torch.tensor(input_ids).unsqueeze(0)
mask = (source != PAD_INDEX).type(torch.uint8) 
source = source.to(device)
mask = mask.to(device)
output = model(source, attention_mask=mask) #, attention_mask=mask
output = output.argmax(axis = 1).cpu().detach().numpy()

if output[0] ==0:
  print("Our Model thinks that your tweet is: not a hate speach")
else:
  print("Our Model thinks that your tweet is: a hate speach")





