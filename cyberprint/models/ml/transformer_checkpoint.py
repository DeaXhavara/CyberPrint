

from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

# Path to your saved checkpoint
CHECKPOINT_PATH = "cyberprint/models/ml/transformer_checkpoint"

def load_checkpoint():
    """
    Load the fine-tuned transformer model and tokenizer from local checkpoint.
    """
    model = DebertaV2ForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
    tokenizer = DebertaV2Tokenizer.from_pretrained(CHECKPOINT_PATH)
    return model, tokenizer

if __name__ == "__main__":
    # Quick test to confirm loading works
    model, tokenizer = load_checkpoint()
    print("✅ Transformer checkpoint loaded successfully from:", CHECKPOINT_PATH)