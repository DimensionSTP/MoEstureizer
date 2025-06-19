from typing import Dict, Any, List
import os

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
load_dataset = datasets.load_dataset


class StructuralDataset:
    def __init__(
        self,
        data_path: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        instruction_column_name: str,
        data_column_name: str,
        chosen_column_name: str,
        rejected_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
    ) -> None:
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.instruction_column_name = instruction_column_name
        self.data_column_name = data_column_name
        self.chosen_column_name = chosen_column_name
        self.rejected_column_name = rejected_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name

    def __call__(self) -> Dict[str, HFDataset]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

        dataset = load_dataset(
            self.dataset_format,
            data_files=full_data_path,
        )["train"]

        output_column_names = [
            "chosen",
            "rejected",
        ]
        input_column_names = [
            self.instruction_column_name,
            self.data_column_name,
            self.chosen_column_name,
            self.rejected_column_name,
        ]
        remove_columns = [
            name for name in input_column_names if name not in output_column_names
        ]

        dataset = dataset.map(
            self.create_conversations,
            batched=True,
            remove_columns=remove_columns,
        )

        split_dataset = dataset.train_test_split(
            test_size=self.split_ratio,
            seed=self.seed,
        )

        if self.is_strict_split:
            train_dataset = split_dataset["train"]
        else:
            train_dataset = dataset

        val_dataset = split_dataset["test"]

        return {
            "train": train_dataset,
            "val": val_dataset,
        }

    def apply_conversation_template(
        self,
        instruction: str,
        data: str,
        label: str,
    ) -> str:
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
        return conversation

    def create_conversations(
        self,
        examples: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        chosen_conversations = []
        rejected_conversations = []

        for i in range(len(examples[self.instruction_column_name])):
            chosen_conversations.append(
                self.apply_conversation_template(
                    instruction=examples[self.instruction_column_name][i],
                    data=examples[self.data_column_name][i],
                    label=examples[self.chosen_column_name][i],
                )
            )
            rejected_conversations.append(
                self.apply_conversation_template(
                    instruction=examples[self.instruction_column_name][i],
                    data=examples[self.data_column_name][i],
                    label=examples[self.rejected_column_name][i],
                )
            )

        return {
            "chosen": chosen_conversations,
            "rejected": rejected_conversations,
        }


class ConversationalDataset:
    def __init__(
        self,
        data_path: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        chosen_column_name: str,
        rejected_column_name: str,
    ) -> None:
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.chosen_column_name = chosen_column_name
        self.rejected_column_name = rejected_column_name

    def __call__(self) -> Dict[str, HFDataset]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

        dataset = load_dataset(
            self.dataset_format,
            data_files=full_data_path,
        )["train"]

        if self.chosen_column_name != "chosen":
            dataset = dataset.rename_column(
                self.chosen_column_name,
                "chosen",
            )
        if self.rejected_column_name != "rejected":
            dataset = dataset.rename_column(
                self.rejected_column_name,
                "rejected",
            )

        output_column_names = [
            "chosen",
            "rejected",
        ]
        remove_columns = [
            column
            for column in dataset.column_names
            if column not in output_column_names
        ]

        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)

        split_dataset = dataset.train_test_split(
            test_size=self.split_ratio,
            seed=self.seed,
        )

        if self.is_strict_split:
            train_dataset = split_dataset["train"]
        else:
            train_dataset = dataset

        val_dataset = split_dataset["test"]

        return {
            "train": train_dataset,
            "val": val_dataset,
        }
