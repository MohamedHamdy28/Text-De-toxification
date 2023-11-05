# Import necessary libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForLanguageModeling, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)

import sys
# Append the directory of the `metric` module to sys.path
sys.path.append('../metric')

# Now you can import your module
from style_trasnfer_accuracy import StyleTransferAccuracy




# Define constants
MAX_LEN = 128  # Maximum sequence length for tokenization

# Load dataset
DATA_FILES = {
    "train": r"../../data/interim/training.csv",
    "test": r"../../data/interim/testing.csv"
}
toxic_dataset = load_dataset("csv", data_files=DATA_FILES)

# Split the training dataset into training and validation sets
train_vald_dataset = toxic_dataset["train"].train_test_split(train_size=0.20, test_size=0.04, seed=20)
train_vald_dataset["validation"] = train_vald_dataset.pop("test")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("rajistics/informal_formal_style_transfer")

# Define preprocessing function for tokenizing the data
def preprocessing_function(examples):
    inputs = examples["toxic"]
    targets = examples["neutral"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=MAX_LEN)
    model_inputs["labels"] = tokenizer(targets, padding="max_length", truncation=True, max_length=MAX_LEN)["input_ids"]
    return model_inputs

# Tokenize datasets
tokenized_datasets = train_vald_dataset.map(
    preprocessing_function,
    batched=True,
    remove_columns=train_vald_dataset["train"].column_names
)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("rajistics/informal_formal_style_transfer")

# Get the total number of layers in the encoder and decoder
total_encoder_layers = len(model.encoder.block)
total_decoder_layers = len(model.decoder.block)

# Calculate the number of layers to freeze (95% in this case)
num_encoder_layers_to_freeze = int(total_encoder_layers * 0.95)
num_decoder_layers_to_freeze = int(total_decoder_layers * 0.95)

# Freeze the specified number of layers in the encoder and decoder
for layer in model.encoder.block[:num_encoder_layers_to_freeze]:
    for param in layer.parameters():
        param.requires_grad = False

for layer in model.decoder.block[:num_decoder_layers_to_freeze]:
    for param in layer.parameters():
        param.requires_grad = False

# Optionally freeze embeddings
for param in model.shared.parameters():
    param.requires_grad = False

# Create data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="T5-detoxification",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    logging_steps=100,
)

# Define custom metric computation
style_transfer_accuracy = StyleTransferAccuracy()

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    style_accuracy = style_transfer_accuracy.classify_preds(batch_size=32, preds=decoded_preds)

    # Calculate the average style accuracy
    average_style_accuracy = sum(style_accuracy) / len(style_accuracy)
    print(average_style_accuracy)
    
    # Return the metric as a dictionary
    return {"average_style_accuracy": average_style_accuracy}

# Create trainer
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the trained model
MODEL_PATH = r"../../models/final_solution"
trainer.save_model(MODEL_PATH)

# Print the location where the model is saved
print(f"Finished training and saved the model in {MODEL_PATH}")
