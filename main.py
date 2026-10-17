import numpy as np
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vertorize import retriever

with Path(__file__).with_name("settings.json").open(encoding="utf-8") as file:
    settings = json.load(file)

# model = OllamaLLM(model="llama2"
model = OllamaLLM(model=settings["model"])

def build_chain():

    template = """
        {command}
    
        Context:
        {context}

        Question: 
        {question}

        Answer:"""

    prompt = ChatPromptTemplate.from_template(template).partial(
        command="\n".join(settings["prompt_type"]["scientific"]["description"])
    )
    chain = (
        {   
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


def main():

    chain = build_chain()
    print("\n--- The Study Expert is ready to answer your questions ---")

    while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                break

            response = chain.invoke(query)
            print(f"\nStudy Expert: {response}")


if __name__ == "__main__":
    main()
