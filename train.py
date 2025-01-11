import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# Load the dataset with `trust_remote_code=True`
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split={"train": "train", "test": "test"}, trust_remote_code=True)

# Print dataset structure to verify
print(dataset["train"])
print(dataset["test"])

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Preprocess dataset
def preprocess_data(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(preprocess_data, remove_columns=["audio", "sentence"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/wav2vec2-finetuned",
    evaluation_strategy="steps",
    save_steps=500,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=3e-4,
    save_total_limit=2,
    logging_dir="./logs"
)

# Define Trainer
trainer = Trainer(
    model=model,
    data_collator=lambda features: processor.pad(features, return_tensors="pt"),
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor
)

# Train the model
trainer.train()

# Save fine-tuned model
model.save_pretrained("./models/wav2vec2-finetuned")
processor.save_pretrained("./models/wav2vec2-finetuned")
