import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_model_path', type=str)
    parser.add_argument('--hub_id', type=str)
    return parser.parse_args()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.local_model_path)
    tokenizer.push_to_hub(args.hub_id)
    model.push_to_hub(args.hub_id)


if __name__ == "__main__":
    args = parse_args()
    main(args)
