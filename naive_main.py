import numpy as np
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vertorize import retriever

# choose the LLM model you want to use for answering questions:
# model = OllamaLLM(model="llama2")
model = OllamaLLM(model="llama3.2:3b")

def main():

    template = """
        You are an technical expert. Use ONLY the following script excerpts to answer.
        If the answer isn't in the context, say "There is no information about this in the original Study scripts."

        Context:
        {context}

        Question: 
        {question}

        Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    print("\n--- The Study Expert is ready to answer your questions ---")

    while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                break

            response = chain.invoke(query)
            print(f"\nStudy Expert: {response}")


if __name__ == "__main__":
    main()