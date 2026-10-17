import numpy as np
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from vertorize import retriever

PROMPT_TEMPLATE = "scientific"

with Path(__file__).with_name("settings.json").open(encoding="utf-8") as file:
    settings = json.load(file)


# --------------------------------------------------
# LLM
# --------------------------------------------------
# model = OllamaLLM(model="llama2"
model = OllamaLLM(model=settings["model"])

# --------------------------------------------------
# Format retrieved documents
# --------------------------------------------------
def format_documents(documents):
    formatted = []
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        source = metadata.get("source", "Unknown PDF")
        chapter = metadata.get("chapter", "Unknown Chapter")
        section = metadata.get("section", "Unknown Section")
        page = metadata.get("page_number", "Unknown")
        text = f"""
                    [Source {index}]
                    PDF: {source}
                    Chapter: {chapter}
                    Section: {section}
                    Page: {page}
                    Content:
                    {doc.page_content}
                """
        formatted.append(text)
    return "\n\n".join(formatted)


def build_chain():

    template = """
        {command}
    
        Context:
        {context}

        Question: 
        {question}

        Answer:"""

    prompt = ChatPromptTemplate.from_template(template).partial(
        command="\n".join(settings["prompt_type"][PROMPT_TEMPLATE]["description"])
    )

    answer_chain = (
        {
            "context": (RunnableLambda(lambda x: x["sources"]) | RunnableLambda(format_documents)),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt
        | model
        | StrOutputParser()
    )

    chain = (
        {   
            "sources": retriever,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(answer=answer_chain)
    )

    return chain




def print_sources(documents):
    print("\n--- Retrieved Sources ---")
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        print(f"\n[Source {index}]")
        print(f"PDF: {metadata.get('source', '?')}")
        print(f"Chapter: {metadata.get('chapter', '?')}")
        print(f"Section: {metadata.get('section', '?')}")
        print(f"Page: {metadata.get('page_number', '?')}")
        print(f"Chunk ID: {metadata.get('chunk_id', '?')}")


def main():

    chain = build_chain()
    print("\n--- The Study Expert is ready to answer your questions ---")

    while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                break

            response = chain.invoke(query)
            print(f"\nStudy Expert: {response['answer']}")
            print_sources(response["sources"])


if __name__ == "__main__":
    main()
