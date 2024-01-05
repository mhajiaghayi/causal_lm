from datasets import load_dataset
import pdb,torch

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") # choose a device
torch.cuda.set_device(device) # set the default device

eli5 = load_dataset("eli5", split="train_asks[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)
print(eli5["train"][0])
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
eli5 = eli5.flatten()
print(eli5["train"][0])
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])
block_size = 128
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)   #This dataset contains the token sequences, but some of these are longer than the maximum input length for the model. 


#You can now use a second preprocessing function to

#concatenate all the sequences
#split the concatenated sequences into shorter chunks defined by block_size, which should be both shorter than the maximum input length and short enough for your GPU RAM.

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
pdb.set_trace()
#Use the end-of-sequence token as the padding token and set mlm=False. This will use the inputs as labels shifted to the right by one element

from transformers import DataCollatorForLanguageModeling
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
#     Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
#     the last epoch before stopping training).
training_args = TrainingArguments(
    output_dir="my_awesome_eli5_clm-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

pdb.set_trace()
# model.save_pretrained("my_awesome_eli5_clm-model")
import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# inference 
prompt = "Somatic hypermutation allows the immune system to"
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("my_awesome_eli5_clm-model")
inputs = inputs.to(device)
model = model.to(device)
inputs = tokenizer(prompt, return_tensors="pt").input_ids

from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("my_awesome_eli5_clm-model")
outputs = model.generate(inputs, max_new_tokens=10, do_sample=True, top_k=50, top_p=0.95,pad_token_id=tokenizer.eos_token_id)
output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(output)
