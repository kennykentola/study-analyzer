# import io
# import json
# import pytest
# from app import app

# @pytest.fixture
# def client():
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         yield client

# def test_analyze_file_no_file(client):
#     response = client.post('/api/files/analyze', data={})
#     assert response.status_code == 400
#     assert b'No file provided' in response.data

# def test_analyze_file_invalid_file(client):
#     data = {
#         'file': (io.BytesIO(b"not a pdf"), 'test.txt')
#     }
#     response = client.post('/api/files/analyze', data=data, content_type='multipart/form-data')
#     assert response.status_code == 400
#     assert b'Invalid PDF file' in response.data

# def test_analyze_file_valid_pdf(client):
#     # Use a minimal valid PDF binary for testing
#     pdf_bytes = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 100 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000179 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n256\n%%EOF'
#     data = {
#         'file': (io.BytesIO(pdf_bytes), 'test.pdf')
#     }
#     response = client.post('/api/files/analyze', data=data, content_type='multipart/form-data')
#     assert response.status_code == 200
#     json_data = json.loads(response.data)
#     assert 'page_count' in json_data
#     assert 'study_time' in json_data
#     assert json_data['page_count'] == 1
#     assert isinstance(json_data['study_time'], (int, float))

# def test_summarize_file_no_file(client):
#     response = client.post('/api/files/summarize', data={})
#     assert response.status_code == 400
#     assert b'No file provided' in response.data

# def test_summarize_file_invalid_file(client):
#     data = {
#         'file': (io.BytesIO(b"not a pdf"), 'test.txt')
#     }
#     response = client.post('/api/files/summarize', data=data, content_type='multipart/form-data')
#     assert response.status_code == 400
#     assert b'Only PDF files are supported' in response.data

# def test_summarize_file_valid_pdf(client):
#     pdf_bytes = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 100 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000179 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n256\n%%EOF'
#     data = {
#         'file': (io.BytesIO(pdf_bytes), 'test.pdf'),
#         'method': 'nltk'
#     }
#     response = client.post('/api/files/summarize', data=data, content_type='multipart/form-data')
#     assert response.status_code == 200
#     json_data = json.loads(response.data)
#     assert 'summary' in json_data
#     assert 'ocr_used' in json_data
#     assert 'page_count' in json_data
#     assert json_data['page_count'] == 1
#     assert isinstance(json_data['summary'], str)



import io
import json
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_analyze_file_no_file(client):
    response = client.post('/api/files/analyze', data={})
    assert response.status_code == 400
    assert b'No file provided' in response.data

def test_analyze_file_invalid_file(client):
    data = {
        'file': (io.BytesIO(b"not a pdf"), 'test.txt')
    }
    response = client.post('/api/files/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'Invalid file type' in response.data

def test_analyze_file_valid_pdf(client):
    pdf_bytes = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 100 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000179 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n256\n%%EOF'
    data = {
        'file': (io.BytesIO(pdf_bytes), 'test.pdf')
    }
    response = client.post('/api/files/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = json.loads(response.data)
    assert 'page_count' in json_data
    assert 'formatted_time' in json_data
    assert json_data['page_count'] == 1
    assert isinstance(json_data['formatted_time'], str)

def test_summarize_file_no_file(client):
    response = client.post('/api/files/summarize', data={})
    assert response.status_code == 400
    assert b'No file provided' in response.data

def test_summarize_file_invalid_file(client):
    data = {
        'file': (io.BytesIO(b"not a pdf"), 'test.txt')
    }
    response = client.post('/api/files/summarize', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'Invalid file type' in response.data

def test_summarize_file_valid_pdf(client):
    pdf_bytes = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 100 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000179 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n256\n%%EOF'
    data = {
        'file': (io.BytesIO(pdf_bytes), 'test.pdf'),
        'num_sentences': '50',  # Updated to match UI
        'summary_type': 'full'
    }
    response = client.post('/api/files/summarize', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = json.loads(response.data)
    assert 'summary' in json_data
    assert 'ocr_used' in json_data
    assert 'page_count' in json_data
    assert json_data['page_count'] == 1
    assert isinstance(json_data['summary'], str)