# DocuMind - Interactive PDF Intelligence

<!-- <div align="center">
  <img src="docs/assets/logo.png" alt="DocuMind Logo" width="200"/>
  <p><strong>A natural language interface for conversing with your PDF documents</strong></p>
</div> -->

---

## 🌟 Overview

DocuMind is an intelligent assistant that enables natural language interaction with PDF documents. Built with Streamlit and powered by Mistral-7B, it allows users to extract insights and information from PDFs through simple conversations.

Simply upload your document and start asking questions - DocuMind will provide concise, direct answers based on the document's content!

<!-- <img src="docs/assets/screenshot.png" alt="DocuMind Screenshot" width="100%"/> -->

## ✨ Features

- 📊 **Intelligent Document Processing**: Extract text from PDFs and generate meaningful responses to queries
- 💬 **Conversational Interface**: Interact with your documents through natural language
- 🎯 **Direct Answer Mode**: Get straight-to-the-point responses without excess context
- 🔍 **Semantic Understanding**: Powered by advanced NLP to comprehend document meaning
- 🎨 **Intuitive UI**: Clean, modern interface for seamless document analysis
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 🔧 Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **NLP Model**: [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (via Hugging Face)
- **Text Processing**: [LangChain](https://python.langchain.com/)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **PDF Extraction**: [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Embeddings**: [sentence-transformers](https://www.sbert.net/)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Hugging Face API Token - [Get one here](https://huggingface.co/settings/tokens)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Ayon128/DocuMind-rag.git
cd DocuMind
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```
HUGGINGFACE_API_TOKEN=your_token_here
```

### Running the App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to access DocuMind.

## 🔍 How It Works

DocuMind processes documents through the following pipeline:

1. **PDF Text Extraction**: Extracts raw text content from uploaded PDFs
2. **Text Chunking**: Splits document into manageable segments
3. **Vector Embedding**: Converts text chunks into numerical vector representations
4. **Storage & Indexing**: Creates a searchable vector database using FAISS
5. **Query Processing**: When you ask a question, it finds the most relevant text chunks
6. **Response Generation**: Uses Mistral-7B to generate concise, accurate answers

## 💻 Usage Guide

1. **Upload a PDF**: Click the upload area or drag and drop your document
2. **Wait for Processing**: DocuMind will extract and index the document content
3. **Ask Questions**: Type natural questions about the document in the chat interface
4. **Review Answers**: Get concise, relevant responses based on the document content
5. **Continue Conversation**: Ask follow-up questions to explore document details further

## 🛠️ Project Structure

```
documind/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (create this)
├── docs/                  # Documentation and assets
│   └── assets/            # Images and screenshots
├── README.md              # Project documentation
└── db_faiss/              # Vector database (generated at runtime)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for hosting the Mistral-7B model
- [Streamlit](https://streamlit.io/) for making Python web apps so simple
- [LangChain](https://python.langchain.com/) for the document processing framework
- All open-source contributors who made this project possible

---

<div align="center">
  <p>📚 Made with ❤️ for document lovers 📚</p>
</div>