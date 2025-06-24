from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
logger = logging.getLogger(__name__)

def fine_tune_qa():
    model_name = "distilbert-base-uncased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Load dataset
    try:
        dataset = load_dataset('json', data_files='qa_dataset.json')['train']
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    def preprocess_function(examples):
        questions = [q.strip() for q in examples['question']]
        contexts = [c.strip() for c in examples['context']]
        answers = [a.strip() for a in examples['answer']]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = inputs.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            context = contexts[i]
            answer = answers[i]
            start_char = context.find(answer)
            if start_char == -1:
                start_positions.append(0)
                end_positions.append(0)
                continue
            end_char = start_char + len(answer)
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./qa_fine_tuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    try:
        logger.info("Starting fine-tuning")
        trainer.train()
        model.save_pretrained("./qa_fine_tuned")
        tokenizer.save_pretrained("./qa_fine_tuned")
        logger.info("Fine-tuning completed and model saved")
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")

if __name__ == "__main__":
    fine_tune_qa()