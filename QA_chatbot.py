






# from transformers import pipeline
# import nltk
# from nltk.tokenize import sent_tokenize
# import logging
# import json
# import os
# import re

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')
# logger = logging.getLogger(__name__)

# LEARNED_ANSWERS_FILE = 'learned_answers.json'

# # Initialize the question-answering pipeline with Flan-T5-large
# try:
#     # qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)  # CPU
#     qa_pipeline = pipeline("question-answering", model="./qa_fine_tuned", tokenizer="./qa_fine_tuned", device=-1)
#     logger.info("Initialized Flan-T5-large for QA")
# except Exception as e:
#     logger.error(f"Failed to load Flan-T5-large: {e}")
#     qa_pipeline = None

# def load_json(file_path, default=None):
#     try:
#         if os.path.exists(file_path):
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 logger.debug(f"Loaded JSON from {file_path}: {data}")
#                 return data
#         logger.warning(f"{file_path} does not exist, returning default: {default}")
#         return default
#     except json.JSONDecodeError as e:
#         logger.error(f"JSON decode error in {file_path}: {e}")
#         return default
#     except Exception as e:
#         logger.error(f"Error loading JSON {file_path}: {e}")
#         return default

# def normalize_text(text):
#     """Normalize text for matching by removing extra spaces and punctuation."""
#     text = re.sub(r'\s+', ' ', text.strip().lower())
#     text = re.sub(r'[^\w\s]', '', text)
#     return text

# def ask_question(question, pdf_cache, filename, section_title=''):
#     """
#     Answer a question based on the provided document content in pdf_cache.
    
#     Args:
#         question (str): The question to answer.
#         pdf_cache (dict): Cache containing document data (text, sentences, sections).
#         filename (str): The name of the file in pdf_cache.
#         section_title (str): Optional section title to limit the context.
    
#     Returns:
#         tuple: (answer, sources)
#             - answer (str): The generated answer.
#             - sources (list): List of source sentences or references.
#     """
#     try:
#         if not question or not filename or filename not in pdf_cache:
#             logger.warning("Invalid question, filename, or missing pdf_cache entry")
#             return "Please provide a valid question and uploaded file.", []

#         # Normalize question for matching
#         normalized_question = normalize_text(question)
#         logger.debug(f"Normalized question: {normalized_question}")

#         # Check learned answers
#         learned_answers = load_json(LEARNED_ANSWERS_FILE, {})
#         for q, a in learned_answers.items():
#             if normalize_text(q) == normalized_question:
#                 logger.info(f"Found learned answer for question: {question}")
#                 return a, ["Learned answer"]

#         # Get context from pdf_cache
#         context = pdf_cache[filename]['text']
#         if section_title:
#             section = next((s for s in pdf_cache[filename]['sections'].values() if s['title'] == section_title), None)
#             if section:
#                 context = section['text']
#                 logger.debug(f"Using section context: {section_title}")
#             else:
#                 logger.warning(f"Section {section_title} not found, using full document")
        
#         if not context.strip():
#             logger.warning("Empty context for question")
#             return "No relevant information found in the document.", []

#         # Tokenize context into sentences
#         sentences = sent_tokenize(context)
#         if not sentences:
#             logger.warning("No sentences found in context")
#             return "No relevant information found.", []

#         # Filter relevant sentences for context
#         query_words = set(word.lower() for word in question.split() if word.isalnum())
#         relevant_sentences = [
#             sent for sent in sentences
#             if query_words & set(word.lower() for word in sent.split() if word.isalnum())
#         ]
#         context = ' '.join(relevant_sentences[:10]) if relevant_sentences else context[:1000]

#         # Prepare input for Flan-T5
#         input_text = f"Question: {question}\nContext: {context[:1000]}"  # Limit context
#         logger.debug(f"Input text for QA: {input_text[:200]}...")

#         # Generate answer
#         if qa_pipeline:
#             response = qa_pipeline(input_text, max_length=150, num_return_sequences=1, temperature=0.7)
#             answer = response[0]['generated_text'].strip()
#             if not answer or answer == input_text or answer.lower().startswith('question:'):
#                 answer = "I couldn't find a specific answer in the provided document. Please provide the correct answer."
#         else:
#             answer = "Question-answering model not available."

#         # Collect sources
#         sources = [sent[:100] + "..." if len(sent) > 100 else sent for sent in relevant_sentences[:3]]

#         logger.info(f"Generated answer: {answer[:100]}...")
#         return answer, sources

#     except Exception as e:
#         logger.error(f"Error answering question: {e}", exc_info=True)
#         return f"Error answering question: {str(e)}", []



# import logging
# import json
# from transformers import pipeline
# import torch

# logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
# logger = logging.getLogger(__name__)

# # Initialize QA pipeline with Flan-T5-base
# qa_pipeline = None
# try:
#     qa_pipeline = pipeline("text2text-generation", model="./qa_fine_tuned", device=-1)
#     logger.info("Loaded fine-tuned Flan-T5-base from ./qa_fine_tuned")
# except Exception as e:
#     logger.warning(f"Failed to load fine-tuned model: {e}. Falling back to google/flan-t5-base")
#     try:
#         qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
#         logger.info("Initialized pre-trained Flan-T5-base for QA")
#     except Exception as e:
#         logger.error(f"Failed to initialize pre-trained Flan-T5-base: {e}")

# def ask_question(question, pdf_cache, filename, section_title=''):
#     try:
#         if not qa_pipeline:
#             return "Error: QA model not available", []

#         # Load learned answers
#         learned_answers = {}
#         try:
#             with open('learned_answers.json', 'r') as f:
#                 learned_answers = json.load(f)
#         except FileNotFoundError:
#             logger.debug("No learned answers file found")
#         except Exception as e:
#             logger.error(f"Error loading learned answers: {e}")

#         question = question.lower().strip()
#         if question in learned_answers:
#             logger.info(f"Answered from learned answers: {question}")
#             return learned_answers[question], ["Learned Answers"]

#         if filename not in pdf_cache:
#             return f"Error: File {filename} not found in cache", []

#         text = pdf_cache[filename]['text']
#         sources = ["Document"]
#         if section_title:
#             sections = pdf_cache[filename]['sections']
#             for idx, section in sections.items():
#                 if section['title'].lower() == section_title.lower():
#                     text = section['text']
#                     sources = [f"Section: {section_title}"]
#                     break

#         context = text  # Use full text (Chroma disabled)
#         logger.info("Using full document text for context")

#         input_text = f"question: {question} context: {context}"
#         with torch.no_grad():  # Reduce memory usage
#             answer = qa_pipeline(input_text, max_length=128)[0]['generated_text']
#         logger.info(f"Generated answer for: {question}")
#         return answer, sources
#     except Exception as e:
#         logger.error(f"Error in ask_question: {e}")
#         return f"Error: {str(e)}", []

# # Clean up memory
# torch.cuda.empty_cache() if torch.cuda.is_available() else None




import logging
import json
from transformers import pipeline
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
logger = logging.getLogger(__name__)

# Initialize FAISS index
dimension = 384  # For 'all-MiniLM-L6-v2'
faiss_index = None
try:
    faiss_index = faiss.IndexFlatIP(dimension)  # Use Inner Product for cosine similarity
    faiss_index = faiss.IndexIDMap(faiss_index)  # Support ID mapping
    logger.info("FAISS index initialized with IndexFlatIP")
except Exception as e:
    logger.error(f"Failed to initialize FAISS index: {e}")
    faiss_index = None

# Initialize sentence transformer
embedder = None
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("SentenceTransformer initialized")
except Exception as e:
    logger.error(f"Failed to initialize SentenceTransformer: {e}")

# Initialize QA pipeline
qa_pipeline = None
try:
    qa_pipeline = pipeline("text2text-generation", model="./qa_fine_tuned", device=0 if torch.cuda.is_available() else -1)
    logger.info("Loaded fine-tuned Flan-T5-base from ./qa_fine_tuned")
except Exception as e:
    logger.warning(f"Failed to load fine-tuned model: {e}. Falling back to google/flan-t5-base")
    try:
        qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if torch.cuda.is_available() else -1)
        logger.info("Initialized pre-trained Flan-T5-base for QA")
    except Exception as e:
        logger.error(f"Failed to initialize pre-trained Flan-T5-base: {e}")

def ask_question(question, pdf_cache, filename, section_title=''):
    try:
        if not qa_pipeline:
            return "Error: QA model not available", []

        learned_answers = {}
        try:
            with open('learned_answers.json', 'r') as f:
                learned_answers = json.load(f)
        except FileNotFoundError:
            logger.debug("No learned answers file found")
        except Exception as e:
            logger.error(f"Error loading learned answers: {e}")

        question = question.lower().strip()
        if question in learned_answers:
            logger.info(f"Answered from learned answers: {question}")
            return learned_answers[question], ["Learned Answers"]

        if filename not in pdf_cache:
            return f"Error: File {filename} not found in cache", []

        text = pdf_cache[filename]['text']
        sources = ["Document"]
        if section_title:
            sections = pdf_cache[filename].get('sections', {})
            for idx, section in sections.items():
                if section['title'].lower() == section_title.lower():
                    text = section['text']
                    sources = [f"Section: {section_title}"]
                    break

        if faiss_index and embedder:
            try:
                sentences = [s for s in text.split('. ') if s.strip()]
                if not sentences:
                    logger.warning(f"No sentences found in text for {filename}")
                    context = text
                else:
                    embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=False)
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize for cosine similarity
                    ids = np.arange(len(sentences), dtype=np.int64)
                    faiss_index.add_with_ids(embeddings, ids)
                    query_embedding = embedder.encode([question], show_progress_bar=False)[0]
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
                    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=3)
                    context = ' '.join([sentences[i] for i in indices[0] if i < len(sentences)])
                    sources.extend([f"Sentence {i+1}" for i in indices[0] if i < len(sentences)])
                    faiss_index.reset()
            except Exception as e:
                logger.error(f"FAISS query failed: {e}")
                context = text
        else:
            logger.info("FAISS disabled, using full document text for context")
            context = text

        input_text = f"question: {question} context: {context}"
        with torch.no_grad():
            answer = qa_pipeline(input_text, max_length=128)[0]['generated_text']
        logger.info(f"Generated answer for: {question}")
        return answer, sources
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return f"Error: {str(e)}", []

torch.cuda.empty_cache() if torch.cuda.is_available() else None