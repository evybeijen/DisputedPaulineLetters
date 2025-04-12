"""
Functions for cleaning, anonymizing, chunking, and lemmatizing Ancient Greek text. 
"""

import unicodedata
import re
from cltk import NLP

# Initialize the CLTK NLP pipeline for Ancient Greek
cltk_nlp = NLP(language="grc")


def remove_accentuation(text):
    """
    Remove diacritical marks (accents) from a string of text.

    Args:
        text (str): Input string to process.

    Returns:
        str: Processed string with accents removed.
    """
    # Normalize text to decomposed form (NFD)
    normalized_text = unicodedata.normalize('NFD', text)
    
    # Remove diacritical marks
    processed_text = ''.join(c for c in normalized_text if not unicodedata.combining(c))
    
    # Recompose text to standard form (NFC)
    processed_text = unicodedata.normalize('NFC', processed_text)
    
    return processed_text
    

def remove_punctuation(text, exceptions=None):
    """
    Removes all punctuation (and non-Greek symbols) from Greek text, keeping only Greek letters, spaces, 
    and optionally specified exception characters.

    Args:
        text (str): Input Greek text.
        exceptions (set[str], optional): Characters to preserve in addition to Greek letters and spaces. Defaults to None.

    Returns:
        str: Cleaned Greek text with only Greek letters, spaces, and specified exception characters.
    """
    if exceptions is None:
        exceptions = set()

    # Regex pattern: only allow Greek letters, spaces, and any specified exceptions
    exceptions_pattern = ''.join(re.escape(char) for char in exceptions)
    allowed_pattern = rf"[^α-ωΑ-Ω\s{exceptions_pattern}]"

    # Remove all characters not in the allowed pattern
    processed_text = re.sub(allowed_pattern, '', text)

    return processed_text


def clean_text(text, exceptions=None):
    """
    Cleans Greek text by removing accentuation, punctuation, capitalization, and excess spaces.

    Args:
        text (str): Input text to clean.
        exceptions (set[str], optional): Characters to preserve in addition to Greek letters and spaces. Defaults to None.

    Returns:
        str: Cleaned text (in lowercase).
    """
    
    # Remove accentuation
    cleaned_text = remove_accentuation(text)
    
    # Remove punctuation
    cleaned_text = remove_punctuation(cleaned_text, exceptions)

    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    
    # Normalize spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


def remove_author_names(text, names_to_remove):
    """
    Replaces specified author names in the text with '<n>'.

    Args:
        text (str): Input text.
        names_to_remove (set[str]): Set of author names to remove.

    Returns:
        str: The processed text with specified names replaced by '<n>'.
    """
    # Create regex pattern to match whole words only, case insensitive
    pattern = r'\b(' + '|'.join(map(re.escape, names_to_remove)) + r')\b'
    
    # Replace matches with '<n>'
    processed_text = re.sub(pattern, '<n>', text, flags=re.IGNORECASE)

    return processed_text


def split_text_into_chunks(text, delimiters={".", ";"}, max_chunk_words=100):
    """
    Splits a text into chunks of consecutive sentences, where each chunk contains no more than `max_chunk_words` words.
    
    Args:
        text (str): Input text to split.
        delimiters (set[str], optional): Set of punctuation marks that delimit sentences. Defaults to {".", ";"}
        max_chunk_words (int, optional): Maximum number of words allowed in each chunk. Defaults to 100.
    
    Returns:
        list: List of text chunks, each with no more than `max_chunk_words` words.
    
    """
    # Split the text into sentences based on the given delimiters 
    sentence_pattern = f"[{''.join(re.escape(d) for d in delimiters)}]"
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ''
    current_words = 0
    
    for sentence in sentences:
        # Clean up any extra spaces from the sentence
        sentence = sentence.strip()
        
        if not sentence:  # Skip empty sentences
            continue
        
        # Count the words in the current sentence
        sentence_words = len(sentence.split())
        
        # If adding the current sentence doesn't exceed max_chunk_words, add it to the current chunk
        if current_words + sentence_words <= max_chunk_words:
            if current_chunk:  # Add a space if this is not the first sentence
                current_chunk += ' '
            current_chunk += sentence
            current_words += sentence_words
        else:
            # If adding the current sentence exceeds max_chunk_words, start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_words = sentence_words
    
    # Append the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def lemmatize_text(text):
    """
    Lemmatizes Ancient Greek text using the CLTK NLP pipeline.

    Args:
        text (str): Input text in Ancient Greek to lemmatize.

    Returns:
        str: A string containing the lemmatized words separated by spaces.
    """
    # Analyze the text and get the lemmatized words
    doc = cltk_nlp.analyze(text=text)
    
    # Create a string of lemmas separated by spaces
    lemmatized_text = " ".join([word.lemma if word.lemma else word.string for word in doc.words])
    
    return lemmatized_text