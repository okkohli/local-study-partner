
import os
import glob
import hashlib

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_metadata import extract_chapter_map


MATERIALS_FOLDER = "./materials"
DB_PATH = "./chroma_chapters"

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 250
ADD_START_INDEX = True
EMBEDDING_BATCH_SIZE = 16


embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

vector_store = Chroma(
    collection_name="study_materials",
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

def initialize_database():

    # Check individual IDs so interrupted indexing can resume safely.
    existing_ids = set(vector_store.get(include=[])["ids"])

    print("Checking for documents to index...")

    documents = []

    pdf_files = sorted(
        glob.glob(os.path.join(MATERIALS_FOLDER, "*.pdf")))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=ADD_START_INDEX)

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file}")

        # Extract chapter structure
        chapter_map = extract_chapter_map(pdf_file)

        # Load PDF pages
        loader = PyPDFLoader(pdf_file)
        pdf_pages = loader.load()

        # Attach metadata
        for page in pdf_pages:
            page_number = page.metadata["page"] + 1
            info = chapter_map.get(page_number, {})
            page.metadata.update({
                "source": os.path.basename(pdf_file),
                "page_number": page_number,
                "chapter": info.get("chapter", "Unknown Chapter"),
                "section": info.get("section", "Unknown Section"),
                "section_path": info.get("section_path", "Unknown")})

        # Split documents
        pdf_chunks = text_splitter.split_documents(pdf_pages)

        # Add stable chunk identifiers
        for chunk in pdf_chunks:
            metadata = chunk.metadata
            chunk_key = (
                f"{metadata['source']}:"
                f"{metadata['page_number']}:"
                f"{metadata.get('start_index', 0)}:"
                f"{chunk.page_content}")

            chunk_id = hashlib.sha256(chunk_key.encode("utf-8")).hexdigest()
            metadata["chunk_id"] = chunk_id

        for chunk in pdf_chunks:
            chunk_id = chunk.metadata["chunk_id"]
            if chunk_id not in existing_ids:
                documents.append(chunk)
                existing_ids.add(chunk_id)
        print(f"Loaded and split {pdf_file} into {len(pdf_chunks)} chunks.")

    if not documents:
        print("No new documents to add.")
        return

    # Limit each Ollama request instead of embedding the entire library at once.
    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch = documents[start:start + EMBEDDING_BATCH_SIZE]
        vector_store.add_documents(
            documents=batch,
            ids=[doc.metadata["chunk_id"] for doc in batch]
        )
        print(f"Indexed {min(start + EMBEDDING_BATCH_SIZE, len(documents))}/{len(documents)} new chunks.")

    print("Documents added to database.")

initialize_database()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
