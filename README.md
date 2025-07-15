# 🎮 Steam Game Review Analyzer

This project provides a Streamlit web application that allows users to search for Steam games, fetch their recent reviews, and then analyze those reviews using a local Large Language Model (LLM) (Ollama with Llama2) to extract key positive and negative feedback points. It also generates word clouds to visualize the most mentioned terms within positive and negative reviews.

## ✨ Features

* **Steam Game Search:** Search for any game available on Steam.

* **Review Fetching:** Retrieve recent reviews for selected games from the Steam API.

* **Sentiment Analysis:** Classify reviews as positive or negative.

* **AI-Powered Summaries:** Utilize a local LLM (Llama2 via Ollama) to generate concise, bulleted summaries of key positive and negative features mentioned in reviews.

* **Word Cloud Visualization:** Generate word clouds highlighting the most prominent terms in both positive and negative review sets, with a focus on sentiment-laden words.

* **Responsive UI:** Built with Streamlit for an interactive and user-friendly web interface.

## 🚀 Technologies Used

* **Python 3.x**

* **Streamlit:** For building the interactive web application.

* **Ollama:** For running local Large Language Models (specifically `llama2`).

* **requests:** For making HTTP requests to the Steam API and fetching stopwords.

* **NLTK (Natural Language Toolkit):** For text preprocessing (tokenization, lemmatization, stopwords).

* **scikit-learn (TfidfVectorizer):** For TF-IDF calculations to identify important terms.

* **wordcloud:** For generating word cloud visualizations.

* **matplotlib:** For plotting the word clouds.

* **Pillow (PIL):** For image processing (used by Streamlit for displaying images).

## ⚙️ Setup and Installation

Follow these steps to get the Steam Review Analyzer running on your local machine.

### Prerequisites

1.  **Python 3.8+:** Ensure you have Python installed.

2.  **Ollama:** Download and install Ollama from [ollama.com](https://ollama.com/).

    * After installation, open your terminal/command prompt and download the `llama2` model:

        ```bash
        ollama run llama2
        ```

        This command will download the model if it's not already present and start a chat session. You can type `bye` to exit the chat. Ensure the Ollama server is running in the background (it usually starts automatically after installation).

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ChetanKuckian/Steam-Game-Reviews.git
    cd Steam-Game-Reviews
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    * **On Windows:**

        ```bash
        .\venv\Scripts\activate
        ```

    * **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **NLTK Data Download:**
    The `review_analyzer.py` script is designed to automatically download necessary NLTK data (`punkt`, `wordnet`, `stopwords`) when `ReviewAnalyzer` is initialized. You do not need to manually run `nltk.download()` unless you encounter issues.

## ▶️ Usage

1.  **Ensure Ollama server is running:**
    Before running the Streamlit app, make sure your Ollama server is active and the `llama2` model is downloaded. You can confirm by running `ollama list` in your terminal.

2.  **Run the Streamlit application:**
    From your project's root directory (where `main.py` is located) and with your virtual environment activated, run:

    ```bash
    streamlit run main.py
    ```

3.  **Interact with the app:**

    * Your web browser will automatically open to the Streamlit application (usually `http://localhost:8501`).

    * Enter a game name in the search bar.

    * Select a game from the dropdown.

    * Click "Analyze Reviews" to get insights and word clouds.

## 📂 Project Structure
.
├── main.py                     # Main Streamlit application file
├── review_analyzer.py          # Core logic for review fetching, analysis, and summary generation
├── requirements.txt            # Python dependencies
└── README.md                   # This file

## 📄 License
This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE)