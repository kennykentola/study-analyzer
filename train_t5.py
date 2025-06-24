# from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
# from datasets import load_dataset
# import json

# # Load dataset
# dataset = load_dataset('json', data_files='qa_dataset.json')
# tokenizer = T5Tokenizer.from_pretrained("t5-base")

# def preprocess_data(examples):
#     inputs = [f"question: {ex['question']} context: {ex['context']}" for ex in examples]
#     targets = [ex['answer'] for ex in examples]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#     labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length").input_ids
#     model_inputs["labels"] = labels
#     return model_inputs

# tokenized_dataset = dataset.map(preprocess_data, batched=True)
# model = T5ForConditionalGeneration.from_pretrained("t5-base")

# training_args = TrainingArguments(
#     output_dir="./t5_finetuned",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=500,
#     save_total_limit=2,
#     logging_steps=100,
#     evaluation_strategy="no",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
# )

# trainer.train()
# model.save_pretrained("./t5_finetuned")
# tokenizer.save_pretrained("./t5_finetuned")





# import logging
# from datasets import Dataset
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
# import torch
# import json

# # Configure logging
# logging.basicConfig(
#     filename='train_t5.log',
#     level=logging.DEBUG,
#     format='%(asctime)s %(levelname)s: %(message)s'
# )

# # Load dataset
# def load_dataset(json_path):
#     try:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         # Expected format: [{"question": str, "context": str, "answer": str}, ...]
#         dataset = Dataset.from_dict({
#             'question': [item['question'] for item in data],
#             'context': [item['context'] for item in data],
#             'answer': [item['answer'] for item in data]
#         })
#         logging.info(f"Loaded dataset with {len(dataset)} examples")
#         return dataset
#     except Exception as e:
#         logging.error(f"Failed to load dataset: {e}")
#         return None

# # Preprocess data for T5
# def preprocess_function(examples, tokenizer):
#     inputs = [f"question: {q} context: {c}" for q, c in zip(examples['question'], examples['context'])]
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
#     labels = tokenizer(examples['answer'], max_length=150, truncation=True, padding='max_length')
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# def main():
#     # Load tokenizer and model
#     try:
#         tokenizer = T5Tokenizer.from_pretrained("t5-small")
#         model = T5ForConditionalGeneration.from_pretrained("t5-small")
#         if torch.cuda.is_available():
#             model.to('cuda')
#             logging.info("Model moved to GPU")
#     except Exception as e:
#         logging.error(f"Failed to load T5 model or tokenizer: {e}")
#         return

#     # Load dataset
#     dataset = load_dataset('qa_dataset.json')  # Replace with your dataset path
#     if not dataset:
#         return

#     # Preprocess dataset
#     tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir='./t5_finetuned',
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         logging_steps=10,
#         save_steps=1000,
#         save_total_limit=2,
#     )

#     # Initialize trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#     )

#     # Train model
#     try:
#         trainer.train()
#         model.save_pretrained('./t5_finetuned')
#         tokenizer.save_pretrained('./t5_finetuned')
#         logging.info("Model fine-tuning completed and saved")
#     except Exception as e:
#         logging.error(f"Training failed: {e}")

# if __name__ == '__main__':
#     main()


# from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
# from datasets import load_dataset
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load dataset
# try:
#     dataset = load_dataset('json', data_files='qa_dataset.json')
#     logger.info("Dataset loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load dataset: {e}")
#     exit(1)

# # Initialize tokenizer
# try:
#     tokenizer = T5Tokenizer.from_pretrained("t5-base")
#     logger.info("T5 tokenizer loaded")
# except Exception as e:
#     logger.error(f"Failed to load tokenizer: {e}")
#     exit(1)

# def preprocess_data(examples):
#     """
#     Preprocess dataset for T5 fine-tuning.
#     """
#     inputs = [f"question: {ex['question']} context: {ex['context']}" for ex in examples]
#     targets = [ex['answer'] for ex in examples]
#     try:
#         model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#         labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length").input_ids
#         model_inputs["labels"] = labels
#         return model_inputs
#     except Exception as e:
#         logger.error(f"Error preprocessing data: {e}")
#         raise

# # Tokenize dataset
# try:
#     tokenized_dataset = dataset.map(preprocess_data, batched=True)
#     logger.info("Dataset tokenized")
# except Exception as e:
#     logger.error(f"Failed to tokenize dataset: {e}")
#     exit(1)

# # Load model
# try:
#     model = T5ForConditionalGeneration.from_pretrained("t5-base")
#     logger.info("T5 model initialized")
# except Exception as e:
#     logger.error(f"Failed to load T5 model: {e}")
#     exit(1)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./t5_finetuned",
#     num_train_epochs=5,
#     per_device_train_batch_size=2,
#     save_steps=50,
#     save_total_limit=2,
#     logging_steps=10,
#     evaluation_strategy="no"
# )

# # Initialize trainer
# try:
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset["train"]
#     )
#     logger.info("Trainer initialized")
# except Exception as e:
#     logger.error(f"Failed to initialize trainer: {e}")
#     exit(1)

# # Train model
# try:
#     trainer.train()
#     logger.info("Training completed")
# except Exception as e:
#     logger.error(f"Training failed: {e}")
#     exit(1)

# # Save model
# try:
#     model.save_pretrained("./t5_finetuned")
#     tokenizer.save_pretrained("./t5_finetuned")
#     logger.info("Model and tokenizer saved to ./t5_finetuned")
# except Exception as e:
#     logger.error(f"Failed to save model: {e}")
#     exit(1)




# import json
# import logging
# from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
# from datasets import Dataset

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s|%(levelname)s|%(message)s',
#     handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # Load dataset
# def load_dataset(file_path='qa_dataset.json'):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         dataset = {
#             'question': [item['question'].strip() for item in data],
#             'answer': [item['answer'].strip() for item in data]
#         }
#         return Dataset.from_dict(dataset)
#     except Exception as e:
#         logger.error(f"Failed to load dataset: {e}")
#         raise

# # Preprocess data
# def preprocess_function(examples, tokenizer):
#     inputs = [f"question: {q} context:" for q in examples['question']]
#     targets = examples['answer']
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
#     labels = tokenizer(targets, max_length=150, truncation=True, padding='max_length')
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# try:
#     # Load model and tokenizer
#     model_name = 't5-base'
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)
#     logger.info("T5 model and tokenizer loaded")

#     # Load and preprocess
#     dataset = load_dataset()
#     tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir='./t5_finetuned',
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         save_steps=50,
#         save_total_limit=1,
#         logging_dir='./logs',
#         logging_steps=10,
#         evaluation_strategy="no",
#         save_strategy="steps",
#     )

#     # Initialize trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#     )

#     # Train
#     logger.info("Starting T5 fine-tuning")
#     trainer.train()
#     model.save_pretrained('./t5_finetuned')
#     tokenizer.save_pretrained('./t5_finetuned')
#     logger.info("Training completed and model saved to ./t5_finetuned")
# except Exception as e:
#     logger.error(f"Training error: {e}", exc_info=True)




# import json
# import logging
# from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
# from datasets import Dataset

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s|%(levelname)s|%(message)s',
#     handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # Load dataset
# def load_dataset(file_path='qa_dataset.json'):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         dataset = {
#             'question': [item['question'].strip() for item in data],
#             'answer': [item['answer'].strip() for item in data]
#         }
#         return Dataset.from_dict(dataset)
#     except Exception as e:
#         logger.error(f"Failed to load dataset: {e}")
#         raise

# # Preprocess data
# def preprocess_function(examples, tokenizer):
#     inputs = [f"question: {q} context:" for q in examples['question']]
#     targets = examples['answer']
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
#     labels = tokenizer(targets, max_length=150, truncation=True, padding='max_length')
#     model_inputs['labels'] = labels['input_ids']
#     return model_inputs

# try:
#     # Load model and tokenizer
#     model_name = 't5-base'
#     tokenizer = T5Tokenizer.from_pretrained(model_name)
#     model = T5ForConditionalGeneration.from_pretrained(model_name)
#     logger.info("T5 model and tokenizer loaded")

#     # Load and preprocess
#     dataset = load_dataset()
#     tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir='./t5_finetuned',
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         save_steps=50,
#         save_total_limit=1,
#         logging_dir='./logs',
#         logging_steps=10,
#         evaluation_strategy="no",
#         save_strategy="steps",
#     )

#     # Initialize trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#     )

#     # Train
#     logger.info("Starting T5 fine-tuning")
#     trainer.train()
#     model.save_pretrained('./t5_finetuned')
#     tokenizer.save_pretrained('./t5_finetuned')
#     logger.info("Training completed and model saved to ./t5_finetuned")
# except Exception as e:
#     logger.error(f"Training error: {e}", exc_info=True)




import json
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s|%(levelname)s|%(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load dataset
def load_dataset(file_path='qa_dataset.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = {
            'question': [item['question'].strip() for item in data],
            'answer': [item['answer'].strip() for item in data]
        }
        return Dataset.from_dict(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

# Preprocess data
def preprocess_function(examples, tokenizer):
    inputs = [f"question: {q} context:" for q in examples['question']]
    targets = examples['answer']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=150, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

try:
    # Load model and tokenizer
    model_name = 'google/flan-t5-large'  ### UPDATE ### Changed base model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    logger.info("Flan-T5-large model and tokenizer loaded")

    # Load and preprocess
    dataset = load_dataset()
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./flan_t5_finetuned',  ### UPDATE ### Changed output directory
        num_train_epochs=3,
        per_device_train_batch_size=2,  ### UPDATE ### Reduced batch size for large model
        save_steps=50,
        save_total_limit=1,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="steps",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    logger.info("Starting Flan-T5-large fine-tuning")
    trainer.train()
    model.save_pretrained('./flan_t5_finetuned')
    tokenizer.save_pretrained('./flan_t5_finetuned')
    logger.info("Training completed and model saved to ./flan_t5_finetuned")
except Exception as e:
    logger.error(f"Training error: {e}", exc_info=True)