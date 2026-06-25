from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return tokenizer, model
