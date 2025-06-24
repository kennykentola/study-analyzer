# import numpy as np
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from nltk_utils import bag_of_words, tokenize, stem
# from model import NeuralNet

# # Load intents
# with open('intents.json', 'r') as f:
#     intents = json.load(f)

# all_words = []
# tags = []
# xy = []
# for intent in intents['intents']:
#     tag = intent['tag']
#     tags.append(tag)
#     for pattern in intent['patterns']:
#         w = tokenize(pattern)
#         all_words.extend(w)
#         xy.append((w, tag))

# ignore_words = ['?', '.', '!']
# all_words = [stem(w) for w in all_words if w not in ignore_words]
# all_words = sorted(set(all_words))
# tags = sorted(set(tags))

# X_train = []
# y_train = []
# for (pattern_sentence, tag) in xy:
#     bag = bag_of_words(pattern_sentence, all_words)
#     X_train.append(bag)
#     label = tags.index(tag)
#     y_train.append(label)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# class ChatDataset(Dataset):
#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.n_samples

# # Hyperparameters
# num_epochs = 1000
# batch_size = 8
# learning_rate = 0.001
# input_size = len(X_train[0])
# hidden_size = 8
# output_size = len(tags)

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)
        
#         outputs = model(words)
#         loss = criterion(outputs, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# print(f'Final loss: {loss.item():.4f}')

# data = {
#     "model_state": model.state_dict(),
#     "input_size": input_size,
#     "hidden_size": hidden_size,
#     "output_size": output_size,
#     "all_words": all_words,
#     "tags": tags
# }

# FILE = "data.pth"
# torch.save(data, FILE)
# print(f'Training complete. Model saved to {FILE}')












# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import load_dataset
# import logging

# logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
# logger = logging.getLogger(__name__)

# def fine_tune_qa():
#     model_name = "google/flan-t5-large"
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     except Exception as e:
#         logger.error(f"Failed to load model or tokenizer: {e}")
#         return

#     # Load dataset
#     try:
#         dataset = load_dataset('json', data_files='qa_dataset.json')['train']
#     except Exception as e:
#         logger.error(f"Failed to load dataset: {e}")
#         return

#     def preprocess_function(examples):
#         questions = [q.strip() for q in examples['question']]
#         answers = [a.strip() for a in examples['answer']]
#         inputs = [f"question: {q} context: {c}" for q, c in zip(questions, examples['context'])]
#         model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#         labels = tokenizer(answers, max_length=128, truncation=True, padding="max_length")["input_ids"]
#         model_inputs["labels"] = labels
#         return model_inputs

#     try:
#         tokenized_dataset = dataset.map(preprocess_function, batched=True)
#     except Exception as e:
#         logger.error(f"Failed to preprocess dataset: {e}")
#         return

#     training_args = TrainingArguments(
#         output_dir="./qa_fine_tuned",
#         evaluation_strategy="no",
#         learning_rate=2e-5,
#         per_device_train_batch_size=4,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_strategy="epoch",
#         logging_dir='./logs',
#         logging_steps=10,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#         tokenizer=tokenizer
#     )

#     try:
#         logger.info("Starting fine-tuning Flan-T5-large")
#         trainer.train()
#         model.save_pretrained("./qa_fine_tuned")
#         tokenizer.save_pretrained("./qa_fine_tuned")
#         logger.info("Fine-tuning completed and model saved to ./qa_fine_tuned")
#     except Exception as e:
#         logger.error(f"Error during fine-tuning: {e}")

# if __name__ == "__main__":
#     fine_tune_qa()


# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import load_dataset
# import logging

# logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
# logger = logging.getLogger(__name__)

# def fine_tune_qa():
#     model_name = "google/flan-t5-base"
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     except Exception as e:
#         logger.error(f"Failed to load model or tokenizer: {e}")
#         return

#     # Load dataset
#     try:
#         dataset = load_dataset('json', data_files='qa_dataset.json')['train']
#     except Exception as e:
#         logger.error(f"Failed to load dataset: {e}")
#         return

#     def preprocess_function(examples):
#         questions = [q.strip() for q in examples['question']]
#         answers = [a.strip() for a in examples['answer']]
#         inputs = [f"question: {q} context: {c}" for q, c in zip(questions, examples['context'])]
#         model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#         labels = tokenizer(answers, max_length=128, truncation=True, padding="max_length")["input_ids"]
#         model_inputs["labels"] = labels
#         return model_inputs

#     try:
#         tokenized_dataset = dataset.map(preprocess_function, batched=True)
#     except Exception as e:
#         logger.error(f"Failed to preprocess dataset: {e}")
#         return

#     training_args = TrainingArguments(
#         output_dir="./qa_fine_tuned",
#         learning_rate=2e-5,
#         per_device_train_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         save_strategy="epoch",
#         logging_dir='./logs',
#         logging_steps=10,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#         tokenizer=tokenizer
#     )

#     try:
#         logger.info("Starting fine-tuning Flan-T5-base")
#         trainer.train()
#         model.save_pretrained("./qa_fine_tuned")
#         tokenizer.save_pretrained("./qa_fine_tuned")
#         logger.info("Fine-tuning completed and model saved to ./qa_fine_tuned")
#     except Exception as e:
#         logger.error(f"Error during fine-tuning: {e}")

# if __name__ == "__main__":
#     fine_tune_qa()




import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
import torch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s|%(levelname)s|%(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_dataset(file_path='qa_dataset.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = {
            'question': [item['question'].strip() for item in data],
            'answer': [item['answer'].strip() for item in data],
            'context': [item['context'].strip() for item in data]  # Include context
        }
        return Dataset.from_dict(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def preprocess_function(examples, tokenizer):
    inputs = [f"question: {q} context: {c}" for q, c in zip(examples['question'], examples['context'])]
    targets = examples['answer']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)  # Dynamic padding
    labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

try:
    # Load model and tokenizer
    model_name = 'google/flan-t5-base'  # Revert to flan-t5-base
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    logger.info("Flan-T5-base model and tokenizer loaded")

    # Load and preprocess dataset
    dataset = load_dataset()
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./qa_fine_tuned',
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Reduced for memory efficiency
        gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="no",  # Use eval_strategy for transformers>=4.40
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        optim="adamw_torch",  # Use AdamW optimizer
        max_grad_norm=1.0,  # Gradient clipping
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train
    logger.info("Starting Flan-T5-base fine-tuning")
    trainer.train()
    model.save_pretrained('./qa_fine_tuned')
    tokenizer.save_pretrained('./qa_fine_tuned')
    logger.info("Training completed and model saved to ./qa_fine_tuned")

    # Clear memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

except Exception as e:
    logger.error(f"Training error: {e}", exc_info=True)