import torch
import json
import re
import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import BitsAndBytesConfig

# Set model repository and device
model_id = "microsoft/phi-2"
# device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_threshold=6.0
)

# Load Phi-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # due to causal language modelling

# Load HellaSwag dataset
# You can also substitute with your own dataset
# dataset = load_dataset("hellaswag", split="validation[:5]")  # For quick demo, use first 100 examples
dataset = load_dataset("tau/commonsense_qa", split="validation")
batch_size = 32
total = len(dataset)

correct_total = 0
correct_list = [] 
k = 0 #gloabl index

for batch_idx in range((total + batch_size - 1) // batch_size):  # ceil division
    start = batch_idx * batch_size
    end = min(start + batch_size, total)
    batch_dataset = dataset.select(range(start,end))

    prompts = []
    answer_keys = []

    for example in batch_dataset:
        # print(example)
        # print(0/0)
        labels = example["choices"]["label"]
        choices_texts = example["choices"]["text"]
        formatted_choices = "".join([f"{label}. {text}\n" for label, text in zip(labels, choices_texts)])
        prompt = (
            f"Question: {example['question']}\n"
            f"Choices:\n{formatted_choices}"
            f"Choose the most suitable option (A, B, C, D, or E) and explain your reasoning."
        )
        prompts.append(prompt)
        answer_keys.append(example["answerKey"])

    # Tokenize batch with padding and truncation
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )

    # Generate outputs
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False
    )

    # Decode generated text and extract predictions
    for i in range(len(outputs)):
        input_length = inputs["input_ids"].shape[1]  # Using shape[1] safely for padded input length
        gen_ids = outputs[i][input_length:]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract choice letter (A-E) from generation
        match = re.search(r"\b([A-E])\b", generated_text)
        predicted = match.group(1) if match else ""

        if predicted == answer_keys[i]:
            correct_list.append(dataset["id"][k])
            correct_total += 1

        k+=1 #gloabl index update

    print(f"Batch {batch_idx + 1}/{(total + batch_size - 1) // batch_size} done.")

# Final accuracy and saving the correct ids
with open('/home/paritosh/soham/reasoning/correct_list.json', 'w') as f:
    json.dump(correct_list, f, indent=4) # indent for pretty printing
accuracy = correct_total / total
print(f"Final accuracy on entire validation set: {accuracy:.4f} ({correct_total}/{total})")