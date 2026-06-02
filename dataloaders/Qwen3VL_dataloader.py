import itertools
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class Qwen3VLIterableDataset(IterableDataset):
    """Iterable dataset that converts image-text examples into Qwen3-VL batches.

    The class intentionally accepts a few common VLM schemas instead of one hard
    coded dataset format.  It works with examples containing `messages`,
    `conversations`, `texts`, `image`/`images`, or `question` + `answer`.
    Dataset-specific column names can be overridden from the YAML config.
    """

    def __init__(
        self,
        data,
        processor,
        batch_size,
        max_length,
        *,
        image_column="images",
        text_column="texts",
        question_column="question",
        answer_column="answer",
        system_prompt=None,
        ignore_visual_tokens=True,
    ):
        super().__init__()
        self.data = data
        self.processor = processor
        self.batch_size = batch_size
        self.max_length = max_length
        self.image_column = image_column
        self.text_column = text_column
        self.question_column = question_column
        self.answer_column = answer_column
        self.system_prompt = system_prompt
        self.ignore_visual_tokens = ignore_visual_tokens

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_data = iter(self.data)
        else:
            iter_data = itertools.islice(self.data, worker_info.id, None, worker_info.num_workers)

        batch = []
        for example in iter_data:
            try:
                batch.append(self._prepare_example(example))
            except Exception:
                continue

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _get_images(self, example):
        images = example.get(self.image_column)
        if images is None and self.image_column != "image":
            images = example.get("image")
        if images is None and self.image_column != "images":
            images = example.get("images")
        return _as_list(images)

    def _prepare_example(self, example):
        images = self._get_images(example)
        messages = self._get_messages(example, has_image=bool(images))
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text, "images": images}

    def _get_messages(self, example, *, has_image):
        raw_messages = (
            example.get("messages")
            or example.get("conversations")
            or example.get(self.text_column)
        )

        if raw_messages is not None:
            messages = self._normalize_messages(raw_messages)
        else:
            question = str(example.get(self.question_column, example.get("text", "")))
            answer = str(example.get(self.answer_column, example.get("caption", "")))
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        if has_image:
            messages = self._insert_image_marker(messages)
        return messages

    def _normalize_messages(self, raw_messages):
        if isinstance(raw_messages, str):
            return [{"role": "user", "content": raw_messages}]

        messages = []
        for item in _as_list(raw_messages):
            if isinstance(item, str):
                role = "assistant" if messages and messages[-1]["role"] == "user" else "user"
                messages.append({"role": role, "content": item})
                continue

            if not isinstance(item, dict):
                continue

            if "role" in item and "content" in item:
                messages.append({"role": item["role"], "content": item["content"]})
                continue

            if "from" in item and "value" in item:
                role = "assistant" if item["from"] in {"gpt", "assistant"} else "user"
                messages.append({"role": role, "content": item["value"]})
                continue

            if "user" in item:
                messages.append({"role": "user", "content": item["user"]})
            if "assistant" in item:
                messages.append({"role": "assistant", "content": item["assistant"]})

        if not messages:
            raise ValueError("Could not normalize VLM conversation example")
        return messages

    @staticmethod
    def _insert_image_marker(messages):
        updated = []
        inserted = False
        for message in messages:
            content = message["content"]
            if not inserted and message["role"] == "user":
                if isinstance(content, list):
                    content = [{"type": "image"}] + content
                else:
                    content = [{"type": "image"}, {"type": "text", "text": str(content)}]
                inserted = True
            updated.append({"role": message["role"], "content": content})
        return updated

    def _format_batch(self, batch):
        texts = [item["text"] for item in batch]
        images_per_sample = [item["images"] for item in batch]
        has_images = any(images_per_sample)

        kwargs = {
            "text": texts,
            "padding": "max_length",
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if has_images:
            kwargs["images"] = [imgs[0] if len(imgs) == 1 else imgs for imgs in images_per_sample]

        encoded = self.processor(**kwargs)
        batch_out = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
        batch_out["labels"] = self._build_labels(batch_out["input_ids"])
        return batch_out

    def _build_labels(self, input_ids):
        labels = input_ids.clone()
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        if self.ignore_visual_tokens:
            special_ids = [
                getattr(self.processor, "image_token_id", None),
                getattr(self.processor, "video_token_id", None),
                getattr(self.processor, "vision_start_token_id", None),
                getattr(self.processor, "vision_end_token_id", None),
            ]
            config = getattr(self.processor, "image_processor", None)
            special_ids.extend([
                getattr(config, "image_token_id", None),
                getattr(config, "video_token_id", None),
            ])
            for token in (
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
            ):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    special_ids.append(token_id)
            for token_id in {x for x in special_ids if x is not None}:
                labels[input_ids == token_id] = -100

        return labels
