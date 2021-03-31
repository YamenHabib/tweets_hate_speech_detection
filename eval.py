import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
from transformers import RobertaTokenizer, RobertaModel

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
  

class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 128)
        self.bn1 = torch.nn.LayerNorm(128)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(128, 2)
        
    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)
        
        return x

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





