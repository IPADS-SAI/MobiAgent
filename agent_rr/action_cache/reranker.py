import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_torch_npu_available, is_torch_cuda_available

class Qwen3Reranker:
    def __init__(self, config):
        path = config.get("path", "Qwen/Qwen3-Reranker-0.6B")
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        device = "cpu"
        if is_torch_cuda_available():
            device = "cuda:0"
        elif is_torch_npu_available():
            device="npu:0"
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map=device).eval()
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.instruct_fmt = config.get("instruct_fmt",
                                       "<Instruct>: Given a phone-use task, retrieve similar tasks that shares at least **{n}** steps with the given task\n<Query>: {query} \n<Document>: {document}")

    def rerank(self, query_tasks, document_task, step):
        input_texts = [self.instruct_fmt.format(n=step, query=query, document=document_task) for query in query_tasks]
        inputs = self.process_inputs(input_texts)
        logits = self.compute_logits(inputs)
        return logits

    def process_inputs(self, input_texts):
        max_length = 8192
        inputs = self.tokenizer(
            input_texts, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
