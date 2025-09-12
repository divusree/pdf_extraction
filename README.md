# pdf_extraction# PDF Extraction

A Streamlit web application for extracting text, tabular and diagrammatic content from PDF files.

## Features

- Upload PDF files through a user-friendly interface
- Extract text content from PDF documents
- View and process PDF content in real-time
- Simple and intuitive user experience

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pdf_extraction.git
cd pdf_extraction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`
3. Upload a PDF file using the interface
4. View the extracted text content

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- Other dependencies listed in `requirements.txt`
