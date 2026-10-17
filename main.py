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

    # text extention to the 'question' for additional commands: 
    text_extention = settings["prompt_type"][PROMPT_TEMPLATE]["text-extention"]

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

    return chain, text_extention


def print_sources(documents):
    output = "\n--- Retrieved Sources ---\n"
    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        output += (
            f"\n[Source {index}]\n"
            f"PDF: {metadata.get('source', '?')}\n"
            f"Chapter: {metadata.get('chapter', '?')}\n"
            f"Section: {metadata.get('section', '?')}\n"
            f"Page: {metadata.get('page_number', '?')}\n"
            f"Chunk ID: {metadata.get('chunk_id', '?')}\n"
        )
    print(output, end="")


def format_source_summary(documents):
    """Format each PDF/chapter/section once with its unique retrieved pages."""
    sources = {}
    for doc in documents:
        metadata = doc.metadata
        source = metadata.get("source") or "Unknown PDF"
        chapter = metadata.get("chapter") or "Unknown Chapter"
        section = metadata.get("section") or "Unknown Section"
        page = metadata.get("page_number")
        if page is None:
            page = "?"

        chapters = sources.setdefault(source, {})
        chapter_info = chapters.setdefault(chapter, {"pages": set(), "sections": {}})
        chapter_info["pages"].add(page)
        chapter_info["sections"].setdefault(section, set()).add(page)

    def format_pages(pages):
        ordered = sorted(
            pages,
            key=lambda page: (0, int(page)) if str(page).isdigit() else (1, str(page)),
        )
        return ", ".join(str(page) for page in ordered)

    output = ""
    for source, chapters in sources.items():
        output += f"\nPDF: {source}\n"
        for chapter, info in chapters.items():
            output += f"  Chapter: {chapter}\n"
            output += f"  Pages: {format_pages(info['pages'])}\n"
            for section, pages in info["sections"].items():
                output += f"    Section: {section} | Pages: {format_pages(pages)}\n"
    return output


def print_source_summary(documents):
    output = "\n--- Source Summary ---\n" + format_source_summary(documents)
    print(output, end="")


def main():

    chain, text_extention = build_chain()
    print("\n--- The Study Expert is ready to answer your questions ---")

    while True:
            input_text = input("\nYou: ")
            query = input_text + text_extention
            if query.lower() in ["exit", "quit"]:
                break

            response = chain.invoke(query)
            print(f"\nStudy Expert: {response['answer']}")
            print_sources(response["sources"])
            print_source_summary(response["sources"])


if __name__ == "__main__":
    main()
