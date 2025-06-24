


from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import json
import logging
import nltk
from werkzeug.utils import secure_filename
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from QA_chatbot import ask_question
import re

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PDF_TEXT_CACHE = {}
LEARNED_ANSWERS_FILE = 'learned_answers.json'
TRANSLATIONS_FILE = 'translations.json'
TASKS_FILE = 'tasks.json'
PREFERENCES_FILE = 'preferences.json'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_json(file_path, default=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.debug(f"Loaded JSON from {file_path}")
            return data
    except FileNotFoundError:
        logger.debug(f"File {file_path} not found, returning default")
        return default if default is not None else {}
    except Exception as e:
        logger.error(f"Error loading JSON {file_path}: {e}")
        return default if default is not None else {}

def save_json(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON {file_path}: {e}")
        raise

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ''
                text += page_text + '\n'
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}, attempting OCR")
                images = convert_from_path(pdf_path)
                text = ''.join(pytesseract.image_to_string(image) for image in images)
                return text, True
            return text, False
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return '', False

def detect_chapters(text):
    chapters = []
    current_chapter = None
    lines = text.split('\n')
    chapter_pattern = re.compile(r'^(Chapter|Section|Part)\s+(\d+|[IVXLCDM]+)\b', re.I)
    for i, line in enumerate(lines):
        match = chapter_pattern.match(line.strip())
        if match:
            if current_chapter:
                chapters.append(current_chapter)
            current_chapter = {
                'title': line.strip(),
                'text': '',
                'start_line': i,
                'end_line': i
            }
        elif current_chapter:
            current_chapter['text'] += line + '\n'
            current_chapter['end_line'] = i
    if current_chapter:
        chapters.append(current_chapter)
    return chapters

def nltk_summarize(text, num_sentences=40):
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return ''
        stop_words = set(nltk.corpus.stopwords.words('english'))
        word_frequencies = {}
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word not in stop_words and word.isalnum():
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1
        sentence_scores = {}
        for sentence in sentences:
            for word, freq in word_frequencies.items():
                if word in sentence.lower():
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        summary = ' '.join(summary_sentences)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return ''

@app.route('/')
def index():
    logger.debug("Serving index.html from templates")
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    logger.debug(f"Serving static file: {path}")
    return send_from_directory('static', path)

@app.route('/api/files/analyze', methods=['POST'])
def analyze_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            text, ocr_used = extract_text_from_pdf(file_path)
            if not text.strip():
                return jsonify({'error': 'No text could be extracted from the PDF'}), 400
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                page_count = len(pdf_reader.pages)
            prefs = load_json(PREFERENCES_FILE, {'time_per_page': 45, 'time_unit': 'minutes'})
            time_per_page = float(prefs.get('time_per_page', 45))
            time_unit = prefs.get('time_unit', 'minutes')
            if time_unit not in ['seconds', 'minutes', 'hours']:
                logger.warning(f"Invalid time_unit '{time_unit}' in preferences, defaulting to 'minutes'")
                time_unit = 'minutes'
                prefs['time_unit'] = time_unit
                save_json(PREFERENCES_FILE, prefs)
            seconds_per_page = time_per_page
            if time_unit == 'minutes':
                seconds_per_page *= 60
            elif time_unit == 'hours':
                seconds_per_page *= 3600
            total_time = page_count * seconds_per_page
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            PDF_TEXT_CACHE[filename] = {
                'text': text,
                'sections': {str(i): {'title': c['title'], 'text': c['text']} for i, c in enumerate(detect_chapters(text))},
                'ocr_used': ocr_used
            }
            logger.info(f"Analyzed file {filename}: {page_count} pages")
            return jsonify({
                'filename': filename,
                'page_count': page_count,
                'formatted_time': formatted_time,
                'time_per_page': time_per_page,
                'time_unit': time_unit
            })
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        return jsonify({'error': str(e)}), 500

# @app.route('/api/files/summarize', methods=['POST'])
# def summarize_file():
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             text, ocr_used = extract_text_from_pdf(file_path)
#             if not text.strip():
#                 return jsonify({'error': 'No text could be extracted from the PDF'}), 400
#             with open(file_path, 'rb') as f:
#                 pdf_reader = PyPDF2.PdfReader(f)
#                 page_count = len(pdf_reader.pages)
#             num_sentences = int(request.form.get('num_sentences', 70))
#             summary_type = request.form.get('summary_type', 'full')
#             prefs = load_json(PREFERENCES_FILE, {'time_per_page': 45, 'time_unit': 'minutes'})
#             time_per_page = float(prefs.get('time_per_page', 45))
#             time_unit = prefs.get('time_unit', 'minutes')
#             if time_unit not in ['seconds', 'minutes', 'hours']:
#                 logger.warning(f"Invalid time_unit '{time_unit}' in preferences, defaulting to 'minutes'")
#                 time_unit = 'minutes'
#                 prefs['time_unit'] = time_unit
#                 save_json(PREFERENCES_FILE, prefs)
#             seconds_per_page = time_per_page
#             if time_unit == 'minutes':
#                 seconds_per_page *= 60
#             elif time_unit == 'hours':
#                 seconds_per_page *= 3600
#             total_time = page_count * seconds_per_page
#             hours, remainder = divmod(total_time, 3600)
#             minutes, seconds = divmod(remainder, 60)
#             formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
#             if summary_type == 'full':
#                 summary = nltk_summarize(text, num_sentences)
#                 PDF_TEXT_CACHE[filename] = {
#                     'text': text,
#                     'sections': {},
#                     'ocr_used': ocr_used
#                 }
#                 logger.info(f"Summarized file {filename}: full summary")
#                 return jsonify({
#                     'filename': filename,
#                     'page_count': page_count,
#                     'formatted_time': formatted_time,
#                     'summary': summary,
#                     'ocr_used': ocr_used
#                 })
#             else:
#                 chapters = detect_chapters(text)
#                 sections = []
#                 for i, chapter in enumerate(chapters):
#                     chapter_summary = nltk_summarize(chapter['text'], num_sentences // max(1, len(chapters)))
#                     sections.append({
#                         'title': chapter['title'],
#                         'summary': chapter_summary,
#                         'start_page': (chapter['start_line'] // 50) + 1,
#                         'end_page': (chapter['end_line'] // 50) + 1,
#                         'subheadings': []
#                     })
#                 PDF_TEXT_CACHE[filename] = {
#                     'text': text,
#                     'sections': {str(i): {'title': c['title'], 'text': c['text']} for i, c in enumerate(chapters)},
#                     'ocr_used': ocr_used
#                 }
#                 logger.info(f"Summarized file {filename}: {len(sections)} sections")
#                 return jsonify({
#                     'filename': filename,
#                     'page_count': page_count,
#                     'formatted_time': formatted_time,
#                     'sections': sections,
#                     'ocr_used': ocr_used
#                 })
#         return jsonify({'error': 'Invalid file type'}), 400
#     except Exception as e:
#         logger.error(f"Error summarizing file: {e}")
#         return jsonify({'error': str(e)}), 500


@app.route('/api/files/summarize', methods=['POST'])
def summarize_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            text, ocr_used = extract_text_from_pdf(file_path)
            if not text.strip():
                return jsonify({'error': 'No text could be extracted from the PDF'}), 400
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                page_count = len(pdf_reader.pages)
            num_sentences = int(request.form.get('num_sentences', 50))  # Default to 50
            summary_type = request.form.get('summary_type', 'full')
            prefs = load_json(PREFERENCES_FILE, {'time_per_page': 45, 'time_unit': 'minutes'})
            time_per_page = float(prefs.get('time_per_page', 45))
            time_unit = prefs.get('time_unit', 'minutes')
            if time_unit not in ['seconds', 'minutes', 'hours']:
                logger.warning(f"Invalid time_unit '{time_unit}' in preferences, defaulting to 'minutes'")
                time_unit = 'minutes'
                prefs['time_unit'] = time_unit
                save_json(PREFERENCES_FILE, prefs)
            seconds_per_page = time_per_page
            if time_unit == 'minutes':
                seconds_per_page *= 60
            elif time_unit == 'hours':
                seconds_per_page *= 3600
            total_time = page_count * seconds_per_page
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            if summary_type == 'full':
                summary = nltk_summarize(text, num_sentences)
                PDF_TEXT_CACHE[filename] = {
                    'text': text,
                    'sections': {},
                    'ocr_used': ocr_used
                }
                logger.info(f"Summarized file {filename}: full summary")
                return jsonify({
                    'filename': filename,
                    'page_count': page_count,
                    'formatted_time': formatted_time,
                    'summary': summary,
                    'ocr_used': ocr_used
                })
            else:
                chapters = detect_chapters(text)
                sections = []
                for i, chapter in enumerate(chapters):
                    chapter_summary = nltk_summarize(chapter['text'], num_sentences // max(1, len(chapters)))
                    sections.append({
                        'title': chapter['title'],
                        'summary': chapter_summary,
                        'start_page': (chapter['start_line'] // 50) + 1,
                        'end_page': (chapter['end_line'] // 50) + 1,
                        'subheadings': []
                    })
                PDF_TEXT_CACHE[filename] = {
                    'text': text,
                    'sections': {str(i): {'title': c['title'], 'text': c['text']} for i, c in enumerate(chapters)},
                    'ocr_used': ocr_used
                }
                logger.info(f"Summarized file {filename}: {len(sections)} sections")
                return jsonify({
                    'filename': filename,
                    'page_count': page_count,
                    'formatted_time': formatted_time,
                    'sections': sections,
                    'ocr_used': ocr_used
                })
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Error summarizing file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        question = data.get('question')
        filename = data.get('filename')
        section_title = data.get('section_title', '')
        if not question or not filename:
            return jsonify({'error': 'Question and filename are required'}), 400
        if filename not in PDF_TEXT_CACHE:
            return jsonify({'error': f'File {filename} not found in cache'}), 400
        answer, sources = ask_question(question, PDF_TEXT_CACHE, filename, section_title)
        logger.info(f"Answered question: {question}")
        return jsonify({'answer': answer, 'sources': sources})
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learn', methods=['POST'])
def learn():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        question = data.get('question')
        answer = data.get('answer')
        if not question or not answer:
            return jsonify({'error': 'Question and answer are required'}), 400
        learned_answers = load_json(LEARNED_ANSWERS_FILE, {})
        learned_answers[question] = answer
        save_json(LEARNED_ANSWERS_FILE, learned_answers)
        logger.info(f"Learned answer for question: {question}")
        return jsonify({'message': 'Answer learned successfully'})
    except Exception as e:
        logger.error(f"Error learning answer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/translations/<lang>')
def get_translations(lang):
    try:
        translations = load_json(TRANSLATIONS_FILE, {})
        return jsonify(translations.get(lang, {}))
    except Exception as e:
        logger.error(f"Error loading translations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preferences', methods=['GET', 'POST'])
def preferences():
    if request.method == 'GET':
        try:
            prefs = load_json(PREFERENCES_FILE, {'time_per_page': 45, 'time_unit': 'minutes'})
            return jsonify(prefs)
        except Exception as e:
            logger.error(f"Error getting preferences: {e}")
            return jsonify({'error': str(e)}), 500
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            time_per_page = data.get('time_per_page')
            time_unit = data.get('time_unit')
            if not isinstance(time_per_page, (int, float)) or time_per_page <= 0:
                return jsonify({'error': 'Time per page must be a positive number'}), 400
            if time_unit not in ['seconds', 'minutes', 'hours']:
                return jsonify({'error': f'Invalid time unit: {time_unit}. Must be seconds, minutes, or hours'}), 400
            seconds_per_page = float(time_per_page)
            if time_unit == 'minutes':
                seconds_per_page *= 60
            elif time_unit == 'hours':
                seconds_per_page *= 3600
            if seconds_per_page > 3600:
                return jsonify({'error': 'Time per page cannot exceed 1 hour in seconds'}), 400
            prefs = {'time_per_page': time_per_page, 'time_unit': time_unit}
            save_json(PREFERENCES_FILE, prefs)
            logger.info(f"Saved preferences: {prefs}")
            return jsonify({'message': 'Preferences saved'})
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks():
    if request.method == 'GET':
        try:
            tasks = load_json(TASKS_FILE, [])
            return jsonify(tasks)
        except Exception as e:
            logger.error(f"Error getting tasks: {e}")
            return jsonify({'error': str(e)}), 500
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            tasks = load_json(TASKS_FILE, [])
            task = {
                'id': len(tasks) + 1,
                'title': data.get('title', ''),
                'description': data.get('description', ''),
                'due_date': data.get('due_date', ''),
                'completed': False
            }
            if not task['title']:
                return jsonify({'error': 'Task title is required'}), 400
            tasks.append(task)
            save_json(TASKS_FILE, tasks)
            logger.info(f"Added task: {task['title']}")
            return jsonify({'message': 'Task added'})
        except Exception as e:
            logger.error(f"Error adding task: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<int:task_id>', methods=['PUT', 'DELETE'])
def task(task_id):
    try:
        tasks = load_json(TASKS_FILE, [])
        task = next((t for t in tasks if t['id'] == task_id), None)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        if request.method == 'PUT':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            task.update({
                'title': data.get('title', task['title']),
                'description': data.get('description', task['description']),
                'due_date': data.get('due_date', task['due_date']),
                'completed': data.get('completed', task['completed'])
            })
            if not task['title']:
                return jsonify({'error': 'Task title is required'}), 400
            save_json(TASKS_FILE, tasks)
            logger.info(f"Updated task {task_id}")
            return jsonify({'message': 'Task updated'})
        elif request.method == 'DELETE':
            tasks = [t for t in tasks if t['id'] != task_id]
            save_json(TASKS_FILE, tasks)
            logger.info(f"Deleted task {task_id}")
            return jsonify({'message': 'Task deleted'})
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)


