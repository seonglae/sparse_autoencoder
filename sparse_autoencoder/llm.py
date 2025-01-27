import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens.utils import tokenize_and_concatenate


def get_acts_iter(
    model_name: str = "gpt2",
    dataset_name: str = "NeelNanda/pile-10k",
    hook_name: str = "blocks.5.hook_resid_pre",
    batch_size: int = 128,
    seed: int = 42,
    device: str = "cuda"
):
    """
    Iteratively yields feature activations extracted via hooks.
    """
    if not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    ds = load_dataset(dataset_name, split="train")
    token_dataset = tokenize_and_concatenate(
        dataset=ds,
        tokenizer=tokenizer,
        streaming=True,
        max_length=128,
        add_bos_token=True,
    )['tokens']

    activation = None

    def hook_fn(_, __, outputs):
        nonlocal activation
        activation = outputs[0]

    block_idx = int(hook_name.split(".")[1])
    handle = model.transformer.h[block_idx].register_forward_hook(hook_fn)

    try:
        for i in range(0, len(token_dataset), batch_size):
            batch_tokens = token_dataset[i:i + batch_size]
            with torch.no_grad():
                _ = model(input_ids=batch_tokens.to(device))
            yield activation
    finally:
        handle.remove()


def get_stats_acts_sample(
    model_name: str = "gpt2",
    dataset_name: str = "NeelNanda/pile-10k",
    hook_name: str = "blocks.5.hook_resid_pre",
    limit: int = 100,
    shuffle: bool = True,
    seed: int = 42,
    device: str = "cuda"
):
    """
    Returns a single tensor (N, num_features) aggregated from the dataset.
    """
    if not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    ds = load_dataset(dataset_name, split="train")
    if shuffle:
        ds = ds.shuffle(seed=seed)
    token_dataset = tokenize_and_concatenate(
        dataset=ds,
        tokenizer=tokenizer,
        streaming=True,
        max_length=128,
        add_bos_token=True,
    )['tokens']
    token_dataset = token_dataset[:limit]

    all_features = []
    def hook_fn(_, __, outputs):
        all_features.append(outputs[0].to("cpu"))

    block_idx = int(hook_name.split(".")[1])
    handle = model.transformer.h[block_idx].register_forward_hook(hook_fn)

    try:
        for tokens in tqdm(token_dataset, desc="Generating stats sample"):
            with torch.no_grad():
                _ = model(input_ids= tokens.to(device))
    finally:
        handle.remove()

    return torch.cat(all_features, dim=0)