import json


def load_full(full_path="open-eqa-v0.json"):
    with open(full_path, "r") as f:
        full_data = json.load(f)
    return full_data


def create_subset(full_data, subset_len):
    output_path = f"open-eqa-{subset_len}.json"
    if subset_len == 557:
        subset_data = [
            x for x in full_data if x["episode_history"].startswith("hm3d-v0/")
        ]
    elif subset_len == 41:
        path_of_3dmem = f"open-eqa-{subset_len}-3dmem.json"
        with open(path_of_3dmem, "r") as f:
            data_of_3dmem = json.load(f)
        ids_of_3dmem = {x["question_id"] for x in data_of_3dmem}
        subset_data = [
            x for x in full_data if x["question_id"] in ids_of_3dmem
        ]
    else:
        raise ValueError(f"Subset length {subset_len} not supported")

    assert len(subset_data) == subset_len

    with open(output_path, "w") as f:
        json.dump(subset_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    full_data = load_full()
    create_subset(full_data, 41)
    create_subset(full_data, 557)
