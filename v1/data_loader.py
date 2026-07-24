from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('gpt2')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)


print(tokenizer.pad_token_id)

def load_data(folder_list, batch_size=4):

    path_patterns = [f'cosmopedia/data/{folder}/train-*.parquet' for folder in folder_list]

    dataset = load_dataset('parquet', data_files=path_patterns, split='train', streaming=True)

    tokenized_dataset = dataset.map(tokenize_function)

    data_collator = DefaultDataCollator(return_tensors='pt')

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )