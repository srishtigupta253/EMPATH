import torch
import numpy as np
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import itertools

model_name = 'tuner007/pegasus_paraphrase'  #Enter any other model
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
  
num_beams = 100
num_return_sequences = 100

my_file = open(r"initial_dataset.txt", "r")
data = my_file.read()
data_into_list = data.split("\n")
print(len(data_into_list))

final_list = []
#final_list.append(data_into_list)
print(len(final_list))
count=0
import numpy as np
import pandas as pd

for context in data_into_list:
  count=count+1
#context = 'You can count on me to support you'
  first_list = get_response(context,num_return_sequences,num_beams)
  first_list = [x.lower() for x in first_list]
  #second_list = []
  final_list.append(first_list)
  first_list = []
  if count%100 == 0:
    print(count)
    paralist = list(itertools.chain(*final_list))
    datasetlist = list(set(paralist))
    print('Dataset so far:',len(datasetlist))
    df = pd.DataFrame(columns = ['text'])
    df['text'] = datasetlist
    df.to_csv('newdata.csv')
