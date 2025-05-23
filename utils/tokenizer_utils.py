# MNIST Digit Classifier (Transformer) - MLX Version
# File: utils/tokenizer_utils.py
# Copyright (c) 2025 Backprop Bunch Team (Yurii, Amy, Guillaume, Aygun)
# Description: Defines vocabulary and tokenization for the decoder output sequence.
# Created: 2025-04-29
# Updated: 2025-04-29


from .logging import logger

# --- Define Special Tokens and Vocabulary ---

# Define constants for special token indices
PAD_TOKEN_ID = 0
START_TOKEN_ID = 1
END_TOKEN_ID = 2
# Optional: UNK_TOKEN_ID = ? (if needed, but maybe not for digit sequences)

# Define the mapping for digits (starting after special tokens)
# Digit '0' corresponds to token ID 3, '1' to 4, ..., '9' to 12
DIGIT_OFFSET = 3

# Full vocabulary dictionary (Token -> ID)
DECODER_VOCAB = {
    "<pad>": PAD_TOKEN_ID,
    "<start>": START_TOKEN_ID,
    "<end>": END_TOKEN_ID,
    **{str(i): i + DIGIT_OFFSET for i in range(10)}  # Digits '0' through '9'
}

# Inverse mapping (ID -> Token) - useful for decoding predictions
ID_TO_DECODER_TOKEN = {v: k for k, v in DECODER_VOCAB.items()}

# Get vocab size dynamically
DECODER_VOCAB_SIZE = len(DECODER_VOCAB)

# --- Helper Functions ---

def labels_to_sequence(
    labels: list[int],
    max_len: int,
    start_token_id: int = START_TOKEN_ID,
    end_token_id: int = END_TOKEN_ID,
    pad_token_id: int = PAD_TOKEN_ID
) -> list[int]:
    """
    Convert a list of digit labels (0-9) into a padded sequence of token IDs.

    Args:
        labels (list[int]): List of digit labels (0-9), e.g., [1, 8, 0, 2].
        max_len (int): Maximum sequence length including start/end tokens.
        start_token_id (int): Token ID for the start token.
        end_token_id (int): Token ID for the end token.
        pad_token_id (int): Token ID for the padding token.

    Returns:
        list[int]: Sequence of token IDs, padded to max_len. Example:
            [START, '1', '8', '0', '2', END, PAD, PAD, ...]
    """
    # Convert digit labels (0-9) to token IDs (3-12)
    token_ids = [label + DIGIT_OFFSET for label in labels]

    # Add start and end tokens
    sequence = [start_token_id] + token_ids + [end_token_id]

    # Pad sequence
    padding_needed = max_len - len(sequence)
    padded_sequence = sequence + [pad_token_id] * padding_needed

    # Truncate if too long (shouldn't happen if max_len is set correctly)
    return padded_sequence[:max_len]

def sequence_to_labels(
    token_ids: list[int],
    start_token_id: int = START_TOKEN_ID,
    end_token_id: int = END_TOKEN_ID,
    pad_token_id: int = PAD_TOKEN_ID
) -> list[int]:
    """
    Convert a sequence of token IDs back to digit labels, stopping at the first
    <end> or <pad> token.

    Args:
        token_ids (list[int]): Sequence of token IDs from model output.
        start_token_id (int): Token ID for the start token.
        end_token_id (int): Token ID for the end token.
        pad_token_id (int): Token ID for the padding token.

    Returns:
        list[int]: Decoded digit labels (0-9) as a list.
    """
    labels = []
    for token_id in token_ids:
        if token_id == start_token_id:
            continue  # Skip start token
        if token_id == end_token_id or token_id == pad_token_id:
            break  # Stop decoding
        digit_label = token_id - DIGIT_OFFSET
        if 0 <= digit_label <= 9:
            labels.append(digit_label)
        else:
            # Handle unexpected tokens if necessary (e.g., log warning)
            pass
    return labels

# --- Test Block ---
if __name__ == "__main__":
    logger.info("--- Decoder Vocabulary ---")
    logger.info(f"Vocab Size: {DECODER_VOCAB_SIZE}")
    logger.info(f"Vocab Dict: {DECODER_VOCAB}")
    logger.info(f"ID to Token: {ID_TO_DECODER_TOKEN}")
    logger.info("-" * 20)

    logger.info("--- Testing label_to_sequence ---")
    example_labels = [1, 8, 0, 2]
    max_len_test = 10  # Example max length
    seq = labels_to_sequence(example_labels, max_len_test)
    logger.info(f"Labels: {example_labels}")
    logger.info(f"Sequence (len={max_len_test}): {seq}")
    assert seq == [1, 4, 11, 3, 5, 2, 0, 0, 0, 0]  # <start>, 1, 8, 0, 2, <end>, <pad> x4
    logger.info("✅ Sequence conversion OK")
    logger.info("-" * 20)

    logger.info("--- Testing sequence_to_labels ---")
    example_sequence = [1, 4, 11, 3, 5, 2, 0, 0, 12, 1]
    decoded_labels = sequence_to_labels(example_sequence)
    logger.info(f"Sequence: {example_sequence}")
    logger.info(f"Decoded Labels: {decoded_labels}")
    assert decoded_labels == [1, 8, 0, 2]
    logger.info("✅ Label decoding OK")
    logger.info("-" * 20)