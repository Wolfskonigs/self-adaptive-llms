import fishfarm
import vllm
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.evalplus import EvalplusTask, load_dataset

from .base import CODE_PROMPT, Task, get_download_dir


class Mbpp2Task(Task):
    def __init__(
        self,
    ):
        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": CODE_PROMPT,
            "mistralai/Mistral-7B-Instruct-v0.3": CODE_PROMPT,
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": CODE_PROMPT, 
        }
        self.system_msg = (
            "You are an exceptionally intelligent coding assistant that "
            " consistently delivers accurate and reliable responses to user "
            "instructions."
        )

        self.target_metric_train = "mbpp_base_pass@1"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = "humaneval_base_pass@1"
        self.has_transfer_split = True
        self.has_training_split = True

    def get_train_data(
        self,
    ):
        train_eval, *test_evals = self.get_evaluator()
        train_data = train_eval.samples
        train_size = len(train_data)
        total_ix = list(range(train_size))
        import random

        random.seed(16)  # fix random seed for reproducibility
        random.shuffle(total_ix)
        train_ix = total_ix[:200]
        valid_ix = total_ix[200:]
        return train_data, train_ix, valid_ix

    def get_rewards(self, res):
        rewards = [1.0 if x["base_correct"] == 1 else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(
        self,
    ):
        res = []
        samples = load_dataset(source_dataset="mbpp")
        res.append(
            EvalplusTask(
                samples[:300],
                context_messages=[
                    fishfarm.Message("system", self.system_msg),
                ],
                source_dataset="mbpp",
            )
        )
        res.append(
            EvalplusTask(
                samples[300:],
                context_messages=[
                    fishfarm.Message("system", self.system_msg),
                ],
                source_dataset="mbpp",
            )
        )
        samples = load_dataset(source_dataset="humaneval")
        res.append(
            EvalplusTask(
                samples,
                context_messages=[
                    fishfarm.Message("system", self.system_msg),
                ],
                source_dataset="humaneval",
            )
        )
        return tuple(res)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}
        user_msg = {"role": "user", "content": samples[ix].instruction}
        assistant_msg = {"role": "assistant", "content": samples[ix].response_prefix}
        return tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg, assistant_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=1024,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="bfloat16",
            download_dir=get_download_dir(),
        )
        chat_template = self.model_to_template[model_id]
        # This may change with vLLM versions.
        m = model.llm_engine.model_executor.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=512,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
