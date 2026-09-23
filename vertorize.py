from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import os
import glob


MATERIALS_FOLDER = "./materials"
DB_PATH = "./chroma_dp"


add_documents = not os.path.exists(DB_PATH)


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
        collection_name="study_materials",
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )


# --------------------------------------------------
# Load PDFs
# --------------------------------------------------
if add_documents:

    print("Adding documents to database...")

    documents = []

    # Find every PDF in materials/
    pdf_files = glob.glob(
        os.path.join(MATERIALS_FOLDER, "*.pdf")
    )


    ## variante-1:
    # pdf_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=2500,
    #     chunk_overlap=250,
    #     add_start_index=True,
    #     separators=["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""]
    # )

    # for pdf_file in pdf_files:
    #     pdf_doc = Document(pdf_file)
    #     pdf_chunks = pdf_splitter.split_documents([pdf_doc])
    #     documents.extend(pdf_chunks)
    #     print(
    #             f"Loaded and split pdf {pdf_file} into {len(pdf_chunks)} chunks."
    #         )

        
    ## variante-2:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        add_start_index=True
    )

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file}")
        loader = PyPDFLoader(pdf_file)
        # PyPDFLoader normally creates one Document per PDF page
        pdf_pages = loader.load()

        # split extracted PDF text into chunks:
        pdf_chunks = text_splitter.split_documents(pdf_pages)
        documents.extend(pdf_chunks)

        print(
            f"Loaded and split {pdf_file} "
            f"into {len(pdf_chunks)} chunks."
        )


    vector_store.add_documents(documents=documents)
    print("Documents added to database.")

else:
    print("Database already exists. Skipping document loading.")

retriever = vector_store.as_retriever(
    search_kwargs={"k":5}
)