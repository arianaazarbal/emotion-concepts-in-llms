import json

import yaml


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def dump_json(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f)


def load_jsonl(file):
    with open(file, "r") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(obj, file):
    with open(file, "w") as f:
        for item in obj:
            f.write(json.dumps(item) + "\n")


def load_yaml(file):
    with open(file, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_yaml(obj, file):
    with open(file, "w") as f:
        yaml.dump(obj, f)
