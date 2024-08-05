# Import necessary libraries
import os
from dotenv import load_dotenv
import numpy as np
import umap
import chromadb
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import matplotlib.pyplot as plt
from quanthub.util import llm

# Import functions from helper_utils
from helper_utils import project_embeddings, word_wrap, extract_text_from_pdf, custom_embedding_function

# Load environment variables
load_dotenv()

def generate_multi_query(query, openai, model="gpt-4"):
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

# Main analysis function
def analyze_pdf(pdf_path, original_query):
    # Set up OpenAI client
    openai = llm.get_llm_client()

    # Read PDF
    pdf_texts = extract_text_from_pdf(pdf_path)

    # Text splitting
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text(pdf_texts)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0,
        tokens_per_chunk=256
    )

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    # Set up ChromaDB
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection(
        "pdf-collection",
        embedding_function=custom_embedding_function
    )

    # Add documents to ChromaDB
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)
    print(f"Number of documents in collection: {chroma_collection.count()}")

    # Generate queries
    aug_queries = generate_multi_query(original_query, openai)
    joint_query = [original_query] + aug_queries

    # Query ChromaDB
    results = chroma_collection.query(
        query_texts=joint_query,
        n_results=5,
        include=["documents", "embeddings"]
    )
    retrieved_documents = results["documents"]

    # Output the results documents
    for i, documents in enumerate(retrieved_documents):
        print(f"Query: {joint_query[i]}")
        print("")
        print("Results:")
        for doc in documents:
            print(word_wrap(doc))
            print("")
        print("-" * 100)

    # Prepare embeddings for visualization
    embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

    original_query_embedding = [custom_embedding_function(original_query)]
    augmented_query_embeddings = [custom_embedding_function(query) for query in joint_query]

    project_original_query = project_embeddings(original_query_embedding, umap_transform)
    project_augmented_queries = project_embeddings(
        augmented_query_embeddings,
        umap_transform
    )

    retrieved_embeddings = results["embeddings"]
    result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]
    projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

    # Visualize embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
        alpha=0.5,
        label="Dataset"
    )
    plt.scatter(
        project_augmented_queries[:, 0],
        project_augmented_queries[:, 1],
        s=150,
        marker="X",
        color="orange",
        label="Augmented Queries"
    )
    plt.scatter(
        projected_result_embeddings[:, 0],
        projected_result_embeddings[:, 1],
        s=100,
        facecolors="none",
        edgecolors="g",
        label="Retrieved Documents"
    )
    plt.scatter(
        project_original_query[:, 0],
        project_original_query[:, 1],
        s=150,
        marker="X",
        color="r",
        label="Original Query"
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"Embedding Visualization: {original_query}")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Set the path to your PDF file
pdf_path = "path/to/your/pdf/file.pdf"

# Set your query
query = "What details can you provide about the factors that led to revenue growth?"

# Run the analysis
analyze_pdf(pdf_path, query)

# You can run additional analyses with different queries by copying the last two lines and modifying the query
# For example:
# new_query = "How has the company's investment in AI impacted its business?"
# analyze_pdf(pdf_path, new_query)