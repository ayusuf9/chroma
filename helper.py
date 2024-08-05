import numpy as np
from pypdf import PdfReader
from quanthub.util import llm

def project_embeddings(embeddings, umap_transform):
    """
    Projects the given embeddings using the provided UMAP transformer.

    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings

def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.

    Args:
    file_path (str): The path to the PDF file.

    Returns:
    str: The extracted text.
    """
    text = []
    with open(file_path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n".join(text)

def custom_embedding_function(text):
    """
    Generates embeddings using the custom OpenAI setup.

    Args:
    text (str): The text to embed.

    Returns:
    list: The embedding vector.
    """
    openai = llm.get_llm_client()
    embedding = openai.Embedding.create(
        input=text,
        deployment_id="text-embedding-ada-002",
        model="text-embedding-ada-002",
        api_key=openai.api_key
    ).data[0].embedding
    return embedding

def generate_multi_query(query, openai, model="gpt-4"):
    """
    Generates multiple related queries based on an original query.

    Args:
    query (str): The original query.
    openai: The OpenAI client.
    model (str): The model to use for query generation.

    Returns:
    list: A list of generated queries.
    """
    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        deployment_id=llm.GPT_4_OMNI_MODEL,
        api_version="2023-07-01-preview",
        api_key=openai.api_key,
        api_base=openai.api_base,
        api_type=openai.api_type,
    )
    content = response.choices[0].message.content
    return content.split("\n")



#!pip install numpy pypdf python-dotenv umap-learn chromadb langchain matplotlib openai sentence-transformers