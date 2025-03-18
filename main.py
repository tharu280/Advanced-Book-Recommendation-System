import pandas as pd
from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


app = FastAPI()


books = pd.read_csv("books_&_emotions.csv")
books["isbn13"] = books["isbn13"].astype(
    str).str.replace(".0", "", regex=False)


hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma(persist_directory="chroma_db",
                  embedding_function=hf_embeddings)


def retrieve_semantic_recommendations(query: str, top_k: int = 10):
    recs = db_books.similarity_search(query, k=top_k)

    books_list = []
    for rec in recs:
        isbn_str = rec.page_content.strip('"').split()[0]
        if isbn_str.endswith(".0"):
            isbn_str = isbn_str[:-2]
        books_list.append(isbn_str)

    return books[books["isbn13"].isin(books_list)].to_dict(orient="records")


@app.get("/recommend")
def recommend_books(query: str, top_k: int = 10):
    recommendations = retrieve_semantic_recommendations(query, top_k)
    return {"recommendations": recommendations}
