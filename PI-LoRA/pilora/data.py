from datasets import load_dataset


def tokenize_dataset(ds, tokenizer, field, max_length=256, tag=""):
    def _fn(batch):
        return tokenizer(batch[field], truncation=True,
                         max_length=max_length, padding="max_length")
    out = ds.map(_fn, batched=True, batch_size=500,
                 remove_columns=ds.column_names, desc=f"Tokenizing {tag}")
    out.set_format("torch")
    return out


def _load_split(spec, split, min_chars=20):
    kwargs = {"path": spec["path"]}
    if spec.get("name"):
        kwargs["name"] = spec["name"]
    ds = load_dataset(**kwargs, split=split)
    field = spec["field"]
    return ds.filter(lambda x: len(str(x[field]).strip()) > min_chars), field


def get_splits(spec, n_train=None, n_valid=None, n_test=None, seed=42):
    """Load train/valid/test; fall back to slicing `train` if a split is missing."""
    def get(name, fallback):
        try:
            return _load_split(spec, name)
        except Exception:
            return _load_split(spec, fallback)

    train, field = get("train", "train[:80%]")
    valid, _ = get("validation", "train[80%:90%]")
    test, _ = get("test", "train[90%:100%]")

    def sub(ds, n):
        if n is None:
            return ds
        return ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    return sub(train, n_train), sub(valid, n_valid), sub(test, n_test), field
