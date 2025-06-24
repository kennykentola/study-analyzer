# from flask import Flask
# app = Flask(__name__)


# import transformers
# import torch
# import nltk
# import PyPDF2
# print("Imports successful")

# @app.route('/')
# def hello():
#     return 'Hello, Flask!'

# if __name__ == '__main__':
#     app.run(debug=True)


from transformers import T5Tokenizer
import logging
logging.basicConfig(level=logging.DEBUG)
try:
    tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned")
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error: {e}")