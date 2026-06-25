from datasets import load_dataset

MAX_LENGTH = 384
DOC_STRIDE = 128

def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset("squad")  # auto-download

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        tokenized = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sample_idx = sample_mapping[i]
            answer = examples["answers"][sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])

                token_start = token_end = None
                for idx, (start, end) in enumerate(offsets):
                    if start <= start_char < end:
                        token_start = idx
                    if start < end_char <= end:
                        token_end = idx
                        break

                if token_start is None or token_end is None:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    start_positions.append(token_start)
                    end_positions.append(token_end)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset
