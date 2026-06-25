# src/summarization/model.py
from transformers import AutoModel
import torch
import torch.nn as nn

class SentenceRanker(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(cls).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())

        return {"loss": loss, "logits": logits}
