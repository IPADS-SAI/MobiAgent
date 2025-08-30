import torch
from sentence_transformers import SentenceTransformer

class Qwen3Embedder:
    def __init__(self, config):
        path = config.get("path", "Qwen/Qwen3-Embedding-0.6B")
        self.model = SentenceTransformer(path)
        self.instruct_fmt = config.get("instruct_fmt",
                                    #    "Instruct: Given a phone-use task, retrieve similar tasks that shares at least **{n}** steps with the given task\nQuery:{query}")
                                       "Instruct: Represent this phone-use task for level **{n}**\nQuery:{query}")

    @torch.no_grad()
    def embed(self, tasks, steps=None):
        if steps is None:
            return self.model.encode(tasks, convert_to_tensor=True, normalize_embeddings=True)
        if len(tasks) != len(steps):
            raise ValueError("Tasks and steps must have the same length")
        input_texts = [self.instruct_fmt.format(n=step, query=task) for task, step in zip(tasks, steps)]
        return self.model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
