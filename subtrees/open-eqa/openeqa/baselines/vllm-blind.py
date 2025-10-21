# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Optional

import tqdm
from openeqa.utils.openai_utils import (
    call_openai_api,
    prepare_openai_messages,
    set_openai_key,
)
from openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    # fmt: off
    parser.add_argument("--key", type=str, default="xxxx")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1") # for vllm
    # fmt: on
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        help="GPT model (default: gpt-4-0613)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="gpt seed (default: 1234)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="gpt temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="gpt maximum tokens (default: 128)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory = args.output_directory / f"AEQA_blind"
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem
        + "-{}-{}.json".format(args.model.replace("/", "--"), args.seed)
    )
    return args


def parse_output(output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


def ask_question(
    question: str,
    openai_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_model: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    openai_seed: int = 1234,
    openai_max_tokens: int = 128,
    openai_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:
    try:
        prompt = load_prompt("blind-llm")
        set_openai_key(key=openai_key)
        messages = prepare_openai_messages(prompt.format(question=question))
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            base_url=openai_base_url,
        )
        return parse_output(output)
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # check for openai api key
    # assert "OPENAI_API_KEY" in os.environ

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = {}
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [k for k, v in results.items()]

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        # generate answer
        question = item["question"]
        answer = ask_question(
            question=question,
            openai_key=args.key,
            openai_base_url=args.base_url,
            openai_model=args.model,
            openai_seed=args.seed,
            openai_max_tokens=args.max_tokens,
            openai_temperature=args.temperature,
            force=args.force,
        )

        # store results
        results[question_id] = answer
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))
    print("Saved to: ", args.output_path)


if __name__ == "__main__":
    main(parse_args())
