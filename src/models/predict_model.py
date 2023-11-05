import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define constants and model path
MODEL_PATH = r"../../models/final_solution"
MAX_LENGTH = 128

# Load the trained model and tokenizer from the specified model path
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Determine the device (GPU or CPU) to use for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the determined device
model.to(device)

# Define the input prompt to be detoxified
prompt = "What the fuck"

# Tokenize the input prompt and prepare the input tensors
# Ensure the tensors are moved to the same device as the model
inputs = tokenizer(
    prompt,
    return_tensors='pt',
    truncation=True,
    max_length=MAX_LENGTH
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Perform inference to generate the detoxified output sequence
# Ensure torch doesn't compute gradients to save memory
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=MAX_LENGTH,
        num_return_sequences=1
    )

# Decode the generated output tensor to get the detoxified text
detoxified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the detoxified text
print(f"Detoxified text: {detoxified_text}")
