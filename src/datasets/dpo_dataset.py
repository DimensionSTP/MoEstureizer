from typing import Dict, Any, List
import os

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class StructuralDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        is_sft: bool,
        is_preprocessed: bool,
        instruction_column_name: str,
        data_column_name: str,
        chosen_column_name: str,
        rejected_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        is_enable_thinking: bool,
        max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.instruction_column_name = instruction_column_name
        self.data_column_name = data_column_name
        self.chosen_column_name = chosen_column_name
        self.rejected_column_name = rejected_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
        self.pretrained_model_name = pretrained_model_name

        if is_preprocessed:
            data_encoder_path = custom_data_encoder_path
        else:
            data_encoder_path = self.pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            data_encoder_path,
            use_fast=True,
        )

        if self.data_encoder.chat_template is None:
            reference_data_encoder = AutoTokenizer.from_pretrained(
                reference_data_encoder_name
            )
            self.data_encoder.chat_template = reference_data_encoder.chat_template

        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id

        if left_padding:
            self.data_encoder.padding_side = "left"
        else:
            self.data_encoder.padding_side = "right"

        self.is_enable_thinking = is_enable_thinking

        dataset = self.get_dataset()
        self.instructions = dataset["instructions"]
        self.datas = dataset["datas"]
        self.choices = dataset["choices"]
        self.rejections = dataset["rejections"]
        self.max_length = max_length

        self.response_start_template = "### Start"
        self.response_start_tokens = self.data_encoder(
            self.response_start_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.response_end_template = "### End"
        self.response_end_tokens = self.data_encoder(
            self.response_end_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt_choice = self.apply_chat_template(
            instruction=self.instructions[idx],
            data=self.datas[idx],
            label=self.choices[idx],
        )
        encoded_choice = self.encode_text(data=prompt_choice)
        if self.is_sft:
            encoded_choice = self.add_sft_label(encoded=encoded_choice)
        if "token_type_ids" in encoded_choice.keys():
            del encoded_choice["token_type_ids"]

        prompt_rejection = self.apply_chat_template(
            instruction=self.instructions[idx],
            data=self.datas[idx],
            label=self.rejections[idx],
        )
        encoded_rejection = self.encode_text(data=prompt_rejection)
        if self.is_sft:
            encoded_rejection = self.add_sft_label(encoded=encoded_rejection)
        if "token_type_ids" in encoded_rejection.keys():
            del encoded_rejection["token_type_ids"]

        choice_start_idx = (
            self.find_pattern_indices(
                input_ids=encoded_choice["input_ids"],
                pattern=self.response_start_tokens,
            )[0]
            if self.is_sft
            else 0
        )

        rejection_start_idx = (
            self.find_pattern_indices(
                input_ids=encoded_rejection["input_ids"],
                pattern=self.response_start_tokens,
            )[0]
            if self.is_sft
            else 0
        )

        prompt_input_ids = encoded_choice["input_ids"][:choice_start_idx]
        prompt_attention_mask = encoded_choice["attention_mask"][:choice_start_idx]

        chosen_input_ids = encoded_choice["input_ids"][choice_start_idx:]
        chosen_attention_mask = encoded_choice["attention_mask"][choice_start_idx:]

        rejected_input_ids = encoded_rejection["input_ids"][rejection_start_idx:]
        rejected_attention_mask = encoded_rejection["attention_mask"][
            rejection_start_idx:
        ]

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            file_name = f"{self.dataset_name}.{self.dataset_format}"
            full_data_path = os.path.join(
                self.data_path,
                file_name,
            )
            data = pd.read_parquet(full_data_path)
            data = data.fillna("_")

            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train" and self.is_strict_split:
                data = train_data
            if self.split == "val":
                data = val_data
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        instructions = (
            data[self.instruction_column_name].apply(lambda x: x.strip()).tolist()
        )
        datas = data[self.data_column_name].apply(lambda x: x.strip()).tolist()
        choices = data[self.chosen_column_name].apply(lambda x: x.strip()).tolist()
        rejections = data[self.rejected_column_name].apply(lambda x: x.strip()).tolist()
        return {
            "instructions": instructions,
            "datas": datas,
            "choices": choices,
            "rejections": rejections,
        }

    def apply_chat_template(
        self,
        instruction: str,
        data: str,
        label: str,
    ) -> str:
        if self.is_sft:
            label = f"\n{self.response_start_template}\n{label}\n{self.response_end_template}\n"

        conversation = [
            {
                self.role_column_name: "system",
                self.content_column_name: instruction,
            },
            {
                self.role_column_name: "user",
                self.content_column_name: data,
            },
            {
                self.role_column_name: self.assistant_column_name,
                self.content_column_name: label,
            },
        ]

        prompt = self.data_encoder.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.is_enable_thinking,
        )
        return prompt

    def encode_text(
        self,
        data: str,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded

    def find_pattern_indices(
        self,
        input_ids: torch.Tensor,
        pattern: torch.Tensor,
    ) -> List[int]:
        pattern_length = pattern.size(0)

        if pattern_length > input_ids.size(0):
            return []

        indices = []
        for i in range(input_ids.size(0) - pattern_length + 1):
            if torch.equal(input_ids[i : i + pattern_length], pattern):
                indices.append(i)
        return indices

    def add_sft_label(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        input_ids = encoded["input_ids"]
        labels = torch.full_like(
            input_ids,
            self.ignore_index,
        )

        start_indices = self.find_pattern_indices(
            input_ids=input_ids,
            pattern=self.response_start_tokens,
        )
        end_indices = self.find_pattern_indices(
            input_ids=input_ids,
            pattern=self.response_end_tokens,
        )

        end_idx_pos = 0
        for start_idx in start_indices:
            content_start = start_idx + len(self.response_start_tokens)

            while end_idx_pos < len(end_indices):
                if end_indices[end_idx_pos] > start_idx:
                    content_end = end_indices[end_idx_pos]
                    labels[content_start:content_end] = input_ids[
                        content_start:content_end
                    ]
                    end_idx_pos += 1
                    break
                end_idx_pos += 1

            else:
                labels[content_start:] = input_ids[content_start:]

        encoded["labels"] = labels
        return encoded


class ConversationalDataset(StructuralDataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        is_sft: bool,
        is_preprocessed: bool,
        chosen_column_name: str,
        rejected_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        pretrained_model_name: str,
        custom_data_encoder_path: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        is_enable_thinking: bool,
        max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.chosen_column_name = chosen_column_name
        self.rejected_column_name = rejected_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
        self.pretrained_model_name = pretrained_model_name

        if is_preprocessed:
            data_encoder_path = custom_data_encoder_path
        else:
            data_encoder_path = self.pretrained_model_name
        self.data_encoder = AutoTokenizer.from_pretrained(
            data_encoder_path,
            use_fast=True,
        )

        if self.data_encoder.chat_template is None:
            reference_data_encoder = AutoTokenizer.from_pretrained(
                reference_data_encoder_name
            )
            self.data_encoder.chat_template = reference_data_encoder.chat_template

        if self.data_encoder.pad_token_id is None:
            self.data_encoder.pad_token_id = self.data_encoder.eos_token_id

        if left_padding:
            self.data_encoder.padding_side = "left"
        else:
            self.data_encoder.padding_side = "right"

        self.is_enable_thinking = is_enable_thinking

        dataset = self.get_dataset()
        self.choices = dataset["choices"]
        self.rejections = dataset["rejections"]
        self.max_length = max_length

        self.response_start_template = "### Start"
        self.response_start_tokens = self.data_encoder(
            self.response_start_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.response_end_template = "### End"
        self.response_end_tokens = self.data_encoder(
            self.response_end_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.choices)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt_choice = self.apply_chat_template(
            conversation=self.choices[idx],
        )
        encoded_choice = self.encode_text(data=prompt_choice)
        if self.is_sft:
            encoded_choice = self.add_sft_label(encoded=encoded_choice)
        if "token_type_ids" in encoded_choice.keys():
            del encoded_choice["token_type_ids"]

        prompt_rejection = self.apply_chat_template(
            conversation=self.rejections[idx],
        )
        encoded_rejection = self.encode_text(data=prompt_rejection)
        if self.is_sft:
            encoded_rejection = self.add_sft_label(encoded=encoded_rejection)
        if "token_type_ids" in encoded_rejection.keys():
            del encoded_rejection["token_type_ids"]

        choice_start_idx = (
            self.find_pattern_indices(
                input_ids=encoded_choice["input_ids"],
                pattern=self.response_start_tokens,
            )[0]
            if self.is_sft
            else 0
        )

        rejection_start_idx = (
            self.find_pattern_indices(
                input_ids=encoded_rejection["input_ids"],
                pattern=self.response_start_tokens,
            )[0]
            if self.is_sft
            else 0
        )

        prompt_input_ids = encoded_choice["input_ids"][:choice_start_idx]
        prompt_attention_mask = encoded_choice["attention_mask"][:choice_start_idx]

        chosen_input_ids = encoded_choice["input_ids"][choice_start_idx:]
        chosen_attention_mask = encoded_choice["attention_mask"][choice_start_idx:]

        rejected_input_ids = encoded_rejection["input_ids"][rejection_start_idx:]
        rejected_attention_mask = encoded_rejection["attention_mask"][
            rejection_start_idx:
        ]

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            file_name = f"{self.dataset_name}.{self.dataset_format}"
            full_data_path = os.path.join(
                self.data_path,
                file_name,
            )
            data = pd.read_parquet(full_data_path)
            data = data.fillna("_")

            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train" and self.is_strict_split:
                data = train_data
            if self.split == "val":
                data = val_data
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        choices = data[self.chosen_column_name].tolist()
        rejections = data[self.rejected_column_name].tolist()
        return {
            "choices": choices,
            "rejections": rejections,
        }

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
    ) -> str:
        preprocessed_conversation = []
        for turn in conversation:
            if (
                turn[self.role_column_name] == self.assistant_column_name
                and self.is_sft
            ):
                content = f"\n{self.response_start_template}\n{turn[self.content_column_name]}\n{self.response_end_template}\n"
                preprocessed_turn = {
                    self.role_column_name: turn[self.role_column_name],
                    self.content_column_name: content,
                }
                preprocessed_conversation.append(preprocessed_turn)
            else:
                preprocessed_turn = {
                    self.role_column_name: turn[self.role_column_name],
                    self.content_column_name: turn[self.content_column_name],
                }
                preprocessed_conversation.append(preprocessed_turn)

        prompt = self.data_encoder.apply_chat_template(
            conversation=preprocessed_conversation,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.is_enable_thinking,
        )
        return prompt
