from model import build_model
from accelerate import Accelerator


def main():
    accelerator = Accelerator()
    model, tok = build_model("configs/models/llama1b.json", accelerator)

    print("Tokenizer vocab size:", tok.vocab_size)
    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
        "M",
    )

    # forward pass test
    text = "hello world"
    inputs = tok(text, return_tensors="pt")
    out = model(**inputs, labels=inputs["input_ids"])
    print("Loss:", out.loss.item())


if __name__ == "__main__":
    main()
