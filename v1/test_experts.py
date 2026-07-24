import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from main import Expert

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

EXPERT_MODEL_PATHS = [
    'models/expert_0_merged.pt', # auto_math_text
    'models/expert_1_merged.pt', # stanford
    'models/expert_2_merged.pt', # stories
    'models/expert_3_merged.pt',# web_samples_v1 (the base model)
    'models/expert_4_merged.pt', # web_samples_v2
    'models/expert_5_merged.pt', # khanacademy, openstax, wikihow
]

PROMPTS = [
    "Question: A train travels at 60 miles per hour. How long does it take to travel 240 miles? \nAnswer:",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including",
    "Once upon a time, in a land filled with enchanted forests and sparkling rivers, there lived a small dragon named Flicker. Unlike the other dragons, Flicker couldn't breathe fire. Instead, he",
    "The internet is a global network of interconnected computers that allows users to share information and communicate with each other. The history of the internet began with the development of",
    "A popular new framework for web development is gaining traction among developers. It focuses on a component-based architecture and offers server-side rendering out of the box. This framework is known as",
    "To properly bake a cake, the first step is to preheat your oven to the correct temperature, typically 350°F (175°C). Next, you should gather all of your ingredients. For a simple vanilla cake, you will need:",
]


tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.eos_token_id


def generate(model, prompt_text, max_length=100, temperature=0.7):
    """
    Generates text from a model given a starting prompt.
    
    Args:
        model: The PyTorch model to use for generation.
        prompt_text (str): The initial text to start generation from.
        max_length (int): The maximum number of tokens to generate.
        temperature (float): Controls the randomness of the output. Higher is more random.
    """
    model.eval()
    
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(DEVICE)
    
    print("\n" + "="*80)
    print(f"PROMPT: '{prompt_text}'")
    print("."*80)
    print("GENERATED TEXT:")
    
    generated_ids = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            attention_mask = torch.ones_like(generated_ids)
            
            outputs = model(generated_ids, attention_mask)
            
            next_token_logits = outputs[:, -1, :]
            
            next_token_logits = next_token_logits / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)
    print("="*80 + "\n")


if __name__ == '__main__':
    for i, model_path in enumerate(EXPERT_MODEL_PATHS):
        print(f"--- Testing Expert {i} ---")
        try:
            expert_model = Expert().to(DEVICE)
            
            state_dict = torch.load(model_path, map_location=DEVICE)
            expert_model.load_state_dict(state_dict)
            
            generate(expert_model, PROMPTS[i])
            
        except FileNotFoundError:
            print(f"ERROR: Model file not found at '{model_path}'. Skipping.")
        except Exception as e:
            print(f"An error occurred while testing expert {i}: {e}")

