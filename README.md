# 📄 InstaDoc: Document Q&A powered by RAG 💡

## Overview ✨

Welcome to **InstaDoc**! This powerful Streamlit application allows you to upload PDF documents and get instant, accurate answers to your questions. InstaDoc offers precise Q&A, detailed document analysis, and seamless interaction, all through a user-friendly interface.

![](assets/image.png)

## 🚀 Features

- **Upload PDFs**: Seamlessly upload your PDF documents and let InstaDoc process them for you.
- **Instant Answers**: Ask questions related to your document and get immediate, accurate responses.
- **Contextual Analysis**: Get relevant information extracted from the document based on your queries.
- **User-Friendly Interface**: Enjoy a smooth and intuitive experience with an easy-to-use chat interface.

## 🛠️ Technologies Used

- **Streamlit**: For building the interactive web application.
- **PyPDF2**: To handle PDF file reading.
- **LangChain**: For advanced document processing and question-answering.
- **Chroma**: For efficient similarity search and vector storage.
- **ChatGroq**: For leveraging the Mixtral-8x7b-32768 model in document analysis.
- **HuggingFaceEmbeddings**: For generating embeddings used in document processing.
- **NLTK**: For natural language processing tasks, specifically sentence tokenization.
- **dotenv**: To manage API keys and environment variables securely.

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/InstaDoc-Intelligent-QnA-Powered-by-RAG.git
   cd InstaDoc-Intelligent-QnA-Powered-by-RAG
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file in the root directory and add your API key:

   ```
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the Streamlit application:**

   ```bash
   streamlit run main.py
   ```

## 🎨 How to Use

1. **Run the Application**:

   ```bash
   streamlit run main.py
   ```

2. **Upload a PDF**:

   - Use the sidebar to upload a PDF document.
   - The application will process the document and create embeddings for it.

3. **Ask Your Questions**:
   - Enter your query related to the document in the chat input.
   - Receive accurate and contextually relevant answers based on the document content.

## 📂 File Structure

- `app.py`: Main application script.
- `requirements.txt`: List of required Python packages.
- `.env`: Environment file for API keys.
- `README.md`: This README file.

## 📋 Features in Detail

### Upload PDFs

Easily upload PDF files using the Streamlit interface. InstaDoc will handle the file processing and prepare it for question-answering.

### Instant Answers

Ask questions about the content of the uploaded document. The system will use advanced language models to provide relevant and accurate answers.

### Contextual Analysis

Get detailed information extracted from your document based on your specific queries.

## 🔧 Troubleshooting

- **Issue**: Application fails to run.
- **Solution**: Ensure all dependencies are installed correctly and the GROQ_API_KEY is set up in the `.env` file.

## 🌟 Contributing

We welcome contributions to improve InstaDoc! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/MohdRasmil7/InstaDoc-Intelligent-QnA-Powered-by-RAG/blob/main/LICENSE) file for details.
