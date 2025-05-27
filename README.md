
# InfoTech Chatbot System

The **InfoBridge InfoTech Assistant** is a retrieval-augmented chatbot system that uses NLP, semantic similarity, and LM Studio-based LLMs to provide intelligent answers based on scraped data from the InfoTechnology website.

---

## System Requirements

- **Python**: 3.11.5  
- **Virtual Environment**: Required (Anaconda recommended)  
- **Python IDE**: VS Code, Jupyter Notebook, or similar

---

## Environment Setup

### 1. Create Virtual Environment

```bash
conda create -n ait526 python=3.11.5 -y
conda activate ait526
```

### 2. Install Dependencies

Extract the project ZIP and run:

```bash
pip install -r requirements.txt
```

---

## LM Studio Setup

Download LM Studio:

- [Windows](https://releases.lmstudio.ai/windows/0.2.31/candidate/b/LM-Studio-0.2.31-Setup.exe)  
- [macOS](https://releases.lmstudio.ai/mac/arm64/0.2.31/2/LM-Studio-0.2.31-arm64.dmg)

Then:
- Launch LM Studio  
- Download the **LLaMA 3.1 8B** model  
- Copy your **API Key** and set it in the code:

```python
openai.api_key = "your-lm-studio-key"
```

---

## How to Run

1. Ensure your JSON data is placed correctly:

```python
chatbot = InteractiveRAGChatBot(data_path="path/to/InfoBridge_scraped_data.json")
```

2. Launch the backend server:

```bash
python app.py
```

3. Open your browser:

```
http://127.0.0.1:5000
```

---

## Frontend Interface

Open `index.html` to launch the UI:

- Chat icon toggles chat window  
- Input box sends queries to `/get_answer`  
- Re-scrape button connects to `/scrape_data`

---

## 🔍 Key Features

| Feature            | Description                                                  |
|--------------------|--------------------------------------------------------------|
| 🔎 Semantic Search | Uses SpaCy + Sentence Transformers                           |
| 🧠 RAG Framework   | Retrieves and generates context-based answers                |
| 💬 Flask API       | Backend routes for `/`, `/get_answer`, `/scrape_data`       |
| 🌐 Web UI          | HTML/JS-based chatbot interface                              |
| 📸 Images          | Displays up to 3 images from the scraped JSON                |
| 💾 Session Memory  | Maintains 5 recent turns for context                         |

---

##  Code Structure

### `app.py`

- Runs Flask app  
- Handles `/`, `/get_answer`, `/scrape_data`

### `InteractiveRAGChatBot` Class

- Extracts keywords  
- Searches JSON for matching chunks  
- Calculates semantic similarity using SentenceTransformer  
- Forms LLM prompt and response using LM Studio API  
- Manages conversation history

---

##  Evaluation Script (Optional)

```python
from sentence_transformers import SentenceTransformer, util

def calculate_similarity(expected, actual):
    emb1 = model.encode(expected, convert_to_tensor=True)
    emb2 = model.encode(actual, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()
```

Set `similarity_threshold = 0.85` to determine correctness.

---

## Web Scraping Script

- Uses Selenium to extract post IDs from:  
  `https://infotechnology.fhwa.dot.gov/bridge/`
- Collects titles, text, and image URLs  
- Saves results to `InfoBridge_scraped_data.json`

---

## Common Issues

| Issue              | Solution                                               |
|--------------------|--------------------------------------------------------|
| LLM not responding | Ensure LM Studio is running and API key is set     |
| JSON file error    | Check path and structure                             |
| No response        | Ensure scraped data contains valid text/images      |

---

## Future Improvements

-  Cloud deployment (e.g., AWS, GCP)  
-  Mobile-friendly UI  
-  Scheduled re-scraping  
-  Better error messages & fallback behavior

---

## References

1. Dong et al. – *LLM fine-tuning and performance* (arXiv:2310.05492)  
2. Perez et al. – *8-bit training for LLMs* (arXiv:2309.17224)  
3. Stack Overflow – [Selenium click scraping](https://stackoverflow.com/questions/71786531/using-selenium-to-click-page-and-scrape-info-from-routed-page)

---

© 2024 – Built for AIT 526 Final Project
