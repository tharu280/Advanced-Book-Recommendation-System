from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from langchain.schema import Document


books = pd.read_csv("books_&_emotions.csv")
books["isbn13"] = books["isbn13"].astype(
    str).str.replace(".0", "", regex=False)


hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


db_books = Chroma(persist_directory="chroma_db",
                  embedding_function=hf_embeddings, collection_name="langchain")


documents = []
for _, row in books.iterrows():
    content = f"{row['isbn13']} {row['title']} {
        row['authors']} {row['categories']} {row['description']}"
    doc = Document(page_content=content, metadata={"isbn13": row["isbn13"]})
    documents.append(doc)


db_books.add_documents(documents)

print(f"Total documents in 'langchain': {
      db_books._client.get_collection('langchain').count()}")
