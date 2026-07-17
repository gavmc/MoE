from data_loader import load_data
from torch import nn
import torch
import math
import os

PADDING_IDX = 50257
VOCAB_SIZE = 50258
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
N_LAYERS = 6
DROPOUT = 0.1

MAX_LR = 3e-4
WARMUP_STEPS = 2000
SAVE_INTERVAL = 10000
NUM_EPOCHS = 1
GRADIENT_CLIP_VAL = 1.0

collections = [
    ['auto_math_text'], # expert 0
    ['stanford'],       # expert 1
    ['stories'],        # expert 2
    ['web_samples_v1'], # expert 3 (Base model)
    ['web_samples_v2'], # expert 4
    ['khanacademy', 'openstax', 'wikihow'], # expert 5
]
BASE_EXPERT_INDEX = 3
BASE_EXPERT_PATH = 'models/expert_3_merged.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PADDING_IDX)
        self.pos_encoder = PositionalEncoding(D_MODEL, DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
            dropout=DROPOUT, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.output_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, input_ids, attention_mask):
        causal_mask = nn.Transformer.generate_square_subsequent_mask(input_ids.size(1), device=device)
        padding_mask = (attention_mask == 0)
        x = self.embedding(input_ids) * math.sqrt(D_MODEL)
        x = self.pos_encoder(x)
        output = self.transformer(src=x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.output_head(output)
        return logits

def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step / WARMUP_STEPS)
    return MAX_LR


if __name__ == "__main__":
    for j in range(len(collections)):
        print(f"\n--- Starting Training for Expert {j} ---")
        print("-" * 50)

        model = Expert().to(device)
        
        if j != BASE_EXPERT_INDEX:
            try:
                base_model_state_dict = torch.load(BASE_EXPERT_PATH, map_location='cpu')
                shared_embedding_weights = {
                    k: v for k, v in base_model_state_dict.items() if 'embedding' in k or 'pos_encoder' in k
                }
                print(f"Loading shared embedding weights from '{BASE_EXPERT_PATH}'.")
                model.load_state_dict(shared_embedding_weights, strict=False)
                
                for name, param in model.named_parameters():
                    if 'embedding' in name or 'pos_encoder' in name:
                        param.requires_grad = False
                print("Embedding layer frozen.")
            except FileNotFoundError:
                print(f"Warning: Base model file not found at '{BASE_EXPERT_PATH}'. Training expert {j} from scratch.")

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
        
        dataloader = load_data(collections[j], batch_size=6)
        os.makedirs(f"checkpoints/expert_{j}", exist_ok=True)
        step = 0
        model.train()
        stop_training = False

        for epoch in range(NUM_EPOCHS):
            for i, batch in enumerate(dataloader):
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                labels = input_ids

                logits = model(input_ids, attention_mask)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = loss_fn(shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRADIENT_CLIP_VAL)
                optimizer.step()

                if step % 100 == 0:
                    print(f"Expert: {j}, Step: {step}, Loss: {loss.item():.4f}, LR: {lr:.6f}")

                # --- CHECKPOINTING ---
                if j == BASE_EXPERT_INDEX and step > WARMUP_STEPS and (step % SAVE_INTERVAL == 0):
                    save_path = f"checkpoints/expert_{j}/step_{step}.pt"
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved checkpoint to {save_path}")
                
                step += 1
                
                # --- STOP CONDITION ---
                stop_steps = 300000 if j == BASE_EXPERT_INDEX else 150000
                if step >= stop_steps:
                    print(f"Finished training Expert {j} after {step} steps.")
                    stop_training = True
                    break
            
            if stop_training:
                break
        
        # Save the final model for this expert
        final_save_path = f"models/expert_{j}_final.pt"
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        torch.save(model.state_dict(), final_save_path)
        print(f"Saved final model for expert {j} to {final_save_path}")

    print("\nAll expert training sessions complete!")
