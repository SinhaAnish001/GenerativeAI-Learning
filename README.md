# GenAI Learning

A hands-on repository documenting my journey into Generative AI using Python and LLM APIs.

## Topics Covered

- Tokenization using `tiktoken`
- API integration with Gemini and Groq
- Prompt engineering and system instructions
- Multi-step LLM workflows
- Structured JSON responses
- Conversation history and repeated API calls

## Tech Stack

- Python
- Google Gemini API
- Groq API
- OpenAI Python SDK (for experiments)
- python-dotenv
- tiktoken

## Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd GenAI-Learning
```

### 2. Create and activate an environment

```bash
conda create -n genai python=3.12
conda activate genai
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root and add the keys for the providers you want to use.

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit your `.env` file or share your API keys.

### 5. Run an experiment

For example:

```bash
python groq_test.py
```

Use the script that matches the experiment you want to run.

## Learning Goals

Build a foundation in LLM APIs, prompt engineering, structured outputs, embeddings, RAG, and AI agents.

## Disclaimer

This repository is for educational purposes. Model availability, API syntax, and free-tier limits can change.
