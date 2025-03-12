# Advanced Semantic Book Recommender

## Overview
The **Semantic Book Recommender** is a book recommendation system that leverages **vector search** with **Hugging Face embeddings** and **ChromaDB** to provide semantic search capabilities. Given a user query, the system retrieves books based on contextual meaning rather than keyword matching.

## Features
- **Semantic Search:** Uses `sentence-transformers/all-MiniLM-L6-v2` to generate book embeddings.
- **Vector Database:** Stores and retrieves book embeddings efficiently using `ChromaDB`.
- **Metadata Filtering:** Allows filtering by category and emotional tone.
- **FastAPI Backend:** Exposes a REST API for integration with frontend applications.

## Tech Stack
- **Python** (Primary Language)
- **FastAPI** (Backend Framework)
- **Hugging Face Transformers** (Embeddings)
- **ChromaDB** (Vector Search)
- **Pandas** (Data Handling)
- **Docker** (Deployment)

## Setup Instructions
### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed.
```sh
pip install -r requirements.txt
```

### 3. Load the Dataset
The system requires a dataset (`books_with_emotions.csv`) with book metadata. Ensure this file is present in the project directory.

### 4. Start the FastAPI Server
Run the FastAPI application:
```sh
uvicorn main:app --reload
```

### 5. API Endpoints
#### Retrieve Recommendations
**GET `/recommendations`**
```sh
curl -X GET "http://127.0.0.1:8000/recommendations?query=A story about adventure&category=Fantasy&tone=Happy"
```
**Response:**
```json
[
    {"title": "The Hobbit", "authors": "J.R.R. Tolkien", "description": "A fantasy adventure..."},
    {"title": "Harry Potter", "authors": "J.K. Rowling", "description": "A magical journey..."}
]
```

## Future Improvements
- Add authentication and user preferences
- Enhance search ranking using fine-tuned models
- Implement UI for better user experience

## License
This project is licensed under the MIT License.

---
Feel free to contribute! 🚀

