from transformers import BertTokenizer, AutoModel
import torch
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")
x = torch.LongTensor(tokenizer.encode('aku adalah anak [MASK]')).view(1,-1)
print("hello")
print(x, model(x)[0].sum())