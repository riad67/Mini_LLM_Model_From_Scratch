"""Train a word-level causal language model and sample a continuation."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F

from transformer_blocks import Block


CORPUS = (
    "hello friends how are you",
    "my name is Riad Hossain",
    "i love traveling and exploring new places",
    "the weather is very cold today",
    "the roads of Dhaka are busy",
    "it is raining in Kishoreganj",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "eid is my favorite festival",
    "new year brings lights and sweets",
    "bangladesh won the cricket match",
)


def encode_corpus(sentences):
    tokens = [word for sentence in sentences for word in (*sentence.split(), "<END>")]
    vocabulary = sorted(set(tokens))
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    data = torch.tensor([word_to_id[word] for word in tokens], dtype=torch.long)
    return data, vocabulary, word_to_id


def get_batch(data, block_size, batch_size=16):
    starts = torch.randint(len(data) - block_size, (batch_size, 1))
    positions = starts + torch.arange(block_size)
    return data[positions], data[positions + 1]


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=6, embedding_dim=32, n_heads=2, n_layers=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *(Block(embedding_dim, block_size, n_heads) for _ in range(n_layers))
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            probabilities = F.softmax(logits[:, -1], dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500, help="training updates (default: 1500)")
    parser.add_argument("--prompt", default="my", help="case-sensitive words from the corpus")
    parser.add_argument("--tokens", type=int, default=15, help="words to sample (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()

    data, vocabulary, word_to_id = encode_corpus(CORPUS)
    prompt = args.prompt.split()
    if args.steps < 1 or args.tokens < 0:
        parser.error("--steps must be positive and --tokens must be nonnegative")
    if not prompt or any(word not in word_to_id for word in prompt):
        parser.error("choose prompt words from the vocabulary: " + " ".join(vocabulary))

    # These small matrix operations run efficiently on a single CPU thread.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = TinyGPT(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"{len(CORPUS)} sentences | {len(data)} tokens | {len(vocabulary)} unique tokens")
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters | CPU | seed {args.seed}")
    for step in range(1, args.steps + 1):
        inputs, targets = get_batch(data, model.block_size)
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 300 == 0 or step == args.steps:
            print(f"Step {step:>4}  loss {loss.item():.4f}")

    model.eval()
    context = torch.tensor([[word_to_id[word] for word in prompt]])
    output = model.generate(context, args.tokens)
    print("\nGenerated text")
    print(" ".join(vocabulary[i] for i in output[0].tolist()))


if __name__ == "__main__":
    main()
