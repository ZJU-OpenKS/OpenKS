import json


def _read_json(input_file, mode="train"):
    lines = []
    with open(input_file, "r") as f:
        for line in f:
            line = json.loads(line.strip())
            text = line["text"]
            label_entities = line.get("label", None)
            words = list(text)
            labels = ["O"] * len(words)
            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert (
                                "".join(words[start_index : end_index + 1])
                                == sub_name
                            )
                            if start_index == end_index:
                                labels[start_index] = "B-" + key
                            else:
                                labels[start_index] = "B-" + key
                                labels[start_index + 1 : end_index + 1] = [
                                    "I-" + key
                                ] * (len(sub_name) - 1)
            lines.append({"words": words, "labels": labels})

    with open(f"/raid/ypj/openSource/cluener_public/{mode}.txt", "w") as f:
        for line in lines:
            for w, l in zip(line["words"], line["labels"]):
                f.write(f"{w}\t{l}\n")
            f.write("\n")


def get_entity_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        # if not isinstance(tag, str):
        #     tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split("-")[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith("I-") and chunk[1] != -1:
            _type = tag.split("-")[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

