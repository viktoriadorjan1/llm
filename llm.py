from typing import List, Tuple
import cohere
import re


def chunk_text(text: str, max_words: int) -> List[str]:
    """
    Divides a large document into chunks of a maximum word length while ensuring sentence integrity.
    The function should avoid breaking sentences across chunks.

    Input:
        text (str): The input text to be chunked.
        max_words (int): Maximum number of words per chunk.

    Output:
        List[str]: A list of string chunks where no chunk exceeds the maximum word limit,
                   and sentences are not split between chunks.
    """

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        # If a sentence is longer than the maximum word limit, allow it to be in a chunk by itself.
        if sentence_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            chunks.append(sentence)
            current_chunk = []
            current_word_count = 0
        # Each chunk should be as close as possible to the maximum word limit without breaking sentences.
        # If adding this sentence would exceed the limit, start a new chunk.
        elif current_word_count + sentence_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = sentence_words
            current_word_count = sentence_word_count
        # Otherwise add sentence to the current chunk.
        else:
            current_chunk.extend(sentence_words)
            current_word_count += sentence_word_count

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def re_rank_chunks(chunks: List[str], query: str) -> List[Tuple[str, float]]:
    """
    Use Cohere’s Re-Ranking API to rank the chunks based on their semantic relevance to a user-provided query.

    Input:
        chunks (List[str]): List of text chunks.
        query (str): The query string for relevance comparison.

    Output:
        List[Tuple[str, float]]: A list of tuples where each tuple contains a chunk and its relevance score,
                                 sorted by highest relevance.
    """

    # Invalid request of API: list of documents must not be empty.
    if not chunks:
        return []

    # Invalid request of API: query must not be empty or be only whitespace.
    if not query.replace(" ", ""):
        return []

    # Initialize the Cohere client.
    cohere_client = cohere.Client("r9KjN9W4jipDsAWHIDd6kNKiPzyLYTO3n40Q5k0I")

    try:
        # Call Cohere's re-rank API.
        response = cohere_client.rerank(query=query, documents=chunks)
        result_items = response.results
    except:
        print("Error using Cohere.")
        return []

    ranked_chunks = [(chunks[item.index], item.relevance_score) for item in result_items]

    # Sort chunks by relevance in descending order.
    ranked_chunks.sort(key=lambda x: x[1], reverse=True)

    return ranked_chunks
