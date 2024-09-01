import argparse
import time

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    start_time = time.time()
    print(f"Start time: {start_time}")

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    import pandas as pd

    # Create a DataFrame with the response and sources
    data = {
        "Header": [query_text],
        "Content": [response_text]
        # "Content": [", ".join(sources)]
    }
    df = pd.DataFrame(data)

    # Export the DataFrame to an .xlsx file
    df.to_excel("output/output.xlsx", index=False)
    end_time = time.time()
    print(f"End time: {end_time}")
    time_taken = end_time - start_time
    print(f"Time taken to export to Excel: {time_taken} seconds")
    return response_text


if __name__ == "__main__":
    main()
