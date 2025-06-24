



import random
import json
import torch
import logging
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from QA_chatbot import ask_question

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load learned knowledge
LEARNED_KNOWLEDGE_FILE = 'learned_knowledge.json'
def load_learned_knowledge():
    try:
        with open(LEARNED_KNOWLEDGE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"questions": []}
    except Exception as e:
        logger.error(f"Error loading learned knowledge: {e}")
        return {"questions": []}

def save_learned_knowledge(data):
    try:
        with open(LEARNED_KNOWLEDGE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug("Learned knowledge saved")
    except Exception as e:
        logger.error(f"Error saving learned knowledge: {e}")

learned_knowledge = load_learned_knowledge()

# Load NeuralNet model
FILE = "data.pth"
try:
    data = torch.load(FILE, map_location=device)
except FileNotFoundError:
    logger.error(f"Model file {FILE} not found. Train the model first.")
    raise
except Exception as e:
    logger.error(f"Error loading model file: {e}")
    raise

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_intent_response(msg):
    try:
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"]), True
        return None, False
    except Exception as e:
        logger.error(f"Error processing intent: {e}")
        return None, False

def get_learned_response(question):
    question = question.lower().strip()
    for item in learned_knowledge['questions']:
        if question == item['question'].lower().strip():
            return item['answer'], True
    return None, False

def get_response(question, pdf_cache=None, filename=None, section_title=''):
    try:
        response, found = get_learned_response(question)
        if found:
            logger.debug(f"Answered from learned knowledge: {question}")
            return response, []

        response, found = get_intent_response(question)
        if found:
            logger.debug(f"Answered from intents: {question}")
            return response, []

        if pdf_cache and filename:
            answer, sources = ask_question(question, pdf_cache, filename, section_title)
            if not answer.startswith("Error"):
                logger.debug(f"Answered from file: {question}")
                return answer, sources

        return (f"I don't know the answer to '{question}'. Please search Google for more information and share the "
                f"correct answer with me so I can learn for next time."), []
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return "Sorry, I encountered an error. Please try again.", []

def learn_new_answer(question, answer):
    try:
        learned_knowledge['questions'].append({"question": question, "answer": answer})
        save_learned_knowledge(learned_knowledge)
        intents['intents'].append({
            "tag": f"learned_{len(learned_knowledge['questions'])}",
            "patterns": [question],
            "responses": [answer]
        })
        with open('intents.json', 'w') as f:
            json.dump(intents, f, indent=4)
        logger.info(f"Learned new answer for question: {question}")
    except Exception as e:
        logger.error(f"Error learning new answer: {e}")

if __name__ == "__main__":
    print("Let's chat! (type 'exit' to quit)")
    while True:
        sentence = input("You: ").strip()
        if sentence.lower() == "exit":
            break
        resp, sources = get_response(sentence)
        print(f"{bot_name}: {resp}")
        if "Please provide the correct answer" in resp:
            answer = input("Please provide the correct answer: ").strip()
            if answer:
                learn_new_answer(sentence, answer)







# import random
# import json
# import torch
# import logging
# from model import NeuralNet
# from nltk_utils import tokenize, bag_of_words
# from QA_chatbot import ask_question
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.utils import embedding_functions

# logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')
# logger = logging.getLogger(__name__)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Initialize sentence transformer and ChromaDB
# try:
#     embedder = SentenceTransformer('all-MiniLM-L6-v2')
#     chroma_client = chromadb.Client()
#     embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
#     collection = chroma_client.get_or_create_collection(
#         name="pdf_chunks",
#         embedding_function=embedding_function
#     )
#     logger.info("Initialized embedder and ChromaDB for chatbot")
# except Exception as e:
#     logger.error(f"Failed to initialize embedder or ChromaDB: {e}")
#     embedder = None
#     collection = None

# # Load intents
# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load learned knowledge
# LEARNED_KNOWLEDGE_FILE = 'learned_knowledge.json'
# def load_learned_knowledge():
#     try:
#         with open(LEARNED_KNOWLEDGE_FILE, 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {"questions": []}
#     except Exception as e:
#         logger.error(f"Error loading learned knowledge: {e}")
#         return {"questions": []}

# def save_learned_knowledge(data):
#     try:
#         with open(data, 'w') as f:
#             json.dump(data, f, indent=4)
#         logger.debug("Learned knowledge saved")
#     except Exception as e:
#         logger.error(f"Error saving learned knowledge: {e}")

# learned_knowledge = load_learned_knowledge()

# # Load NeuralNet model
# FILE = "data.pth"
# try:
#     data = torch.load(FILE, map_location=device)
# except FileNotFoundError:
#     logger.error(f"Model file {FILE} not found. Train the model first.")
#     raise
# except Exception as e:
#     logger.error(f"Error loading model file: {e}")
#     raise

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Sam"

# def get_intent_response(msg):
#     try:
#         sentence = tokenize(msg)
#         X = bag_of_words(sentence, all_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)

#         output = model(X)
#         _, predicted = torch.max(output, dim=1)
#         tag = tags[predicted.item()]

#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         if prob.item() > 0.75:
#             for intent in intents['intents']:
#                 if tag == intent["tag"]:
#                     return random.choice(intent["responses"]), True
#         return None, False
#     except Exception as e:
#         logger.error(f"Error processing intent: {e}")
#         return None, False

# def get_learned_response(question):
#     question = question.lower().strip()
#     for item in learned_knowledge['questions']:
#         if question == item['question'].lower().strip():
#             return item['answer'], True
#     return None, False

# def get_response(question, pdf_cache=None, filename=None, section_title=''):
#     try:
#         response, found = get_learned_response(question)
#         if found:
#             logger.debug(f"Answered from learned knowledge: {question}")
#             return response, []

#         response, found = get_intent_response(question)
#         if found:
#             logger.debug(f"Answered from intents: {question}")
#             return response, []

#         if pdf_cache and filename:
#             answer, sources = ask_question(
#                 question,
#                 pdf_cache,
#                 filename,
#                 section_title,
#                 collection=collection,
#                 embedder=embedder
#             )
#             if not answer.startswith("Error"):
#                 logger.debug(f"Answered from file: {question}")
#                 return answer, sources

#         return (f"I don't know the answer to '{question}'. Please search Google for more information and share the "
#                 f"correct answer with me so I can learn for next time."), []

#     except Exception as e:
#         logger.error(f"Error processing question: {e}")
#         return "Sorry, I encountered an error. Please try again.", []

# def learn_new_answer(question, answer):
#     try:
#         learned_knowledge['questions'].append({"question": question, "answer": answer})
#         save_learned_knowledge(learned_knowledge)
#         intents['intents'].append({
#             "tag": f"learned_{len(learned_knowledge['questions'])}",
#             "patterns": [question],
#             "responses": [answer]
#         })
#         with open('intents.json', 'w') as f:
#             json.dump(intents, f, indent=4)
#         logger.info(f"Learned new answer for question: {question}")
#     except Exception as e:
#         logger.error(f"Error learning new answer: {e}")

# if __name__ == "__main__":
#     print("Let's chat! Type 'exit' to quit")
#     while True:
#         sentence = input("You: ").strip()
#         if sentence.lower() == 'exit':
#             break
#         resp, sources = get_response(sentence)
#         print(f"{bot_name}: {resp}")
#         if "Please provide the correct answer" in resp:
#             answer = input("Please provide the correct answer: ").strip()
#             if answer:
#                 learn_new_answer(sentence, answer)