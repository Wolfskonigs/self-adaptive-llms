import vllm
import wandb
from datasets import load_dataset
from tasks.evaluator import SimpleEvaluator

from .base import Task, get_download_dir

class ChatbotTask(Task):
    def __init__(self):
        self.model_to_template = {
            "meta-llama/Meta-Llama-3-8B-Instruct": "{system}\nUser: {user}\nAssistant: ",
            "mistralai/Mistral-7B-Instruct-v0.3": "{system}\nUser: {user}\nAssistant: ",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "{system}\nUser: {user}\nAssistant: ",
        }
        self.system_msg = (
            "You are a friendly and engaging chatbot that provides relevant, clear, and conversationally engaging responses."
        )
        self.dataset = None  # Store dataset in memory
        self.has_training_split = True  # Add back to satisfy script
        self.has_transfer_split = False  # Add this to prevent AttributeError
        self.load_dataset()

    def load_dataset(self):
        """Load and preprocess the dataset manually."""
        print("Loading dataset from Hugging Face...")
        raw_data = load_dataset("databricks/databricks-dolly-15k")["train"]
        self.dataset = list(raw_data)  # Convert to list for easy indexing
        print(f"Dataset loaded: {len(self.dataset)} samples")

    def get_evaluator(self):
        """Returns evaluator objects instead of dataset splits."""
        print("Returning proper evaluator objects instead of dataset splits.")

        train_size = int(0.8 * len(self.dataset))  # 80% train, 20% test
        train_samples = self.dataset[:train_size]
        test_samples = self.dataset[train_size:]

        train_eval = SimpleEvaluator(train_samples)  # Wrap datasets in evaluators
        test_eval = SimpleEvaluator(test_samples)

        return train_eval, test_eval  # Return tuple
                        
    def get_rewards(self, res):
        """Return a placeholder reward system for now."""
        return [0.0 for _ in res]  # Placeholder rewards (neutral score)

    def get_train_data(self):
        """Loads train/validation split from evaluator's dataset."""

        train_eval, _ = self.get_evaluator()  # Get the evaluator object
        train_samples = train_eval.samples  # Extract dataset from evaluator
        train_size = len(train_samples)

        total_ix = list(range(train_size))
        import random

        random.seed(16)
        random.shuffle(total_ix)

        train_ix = total_ix[: int(0.8 * train_size)]  # 80% Training
        valid_ix = total_ix[int(0.8 * train_size) :]  # 20% Validation

        return train_samples, train_ix, valid_ix 

    def get_prompt(self, tokenizer, samples, ix, model_id):
        """Generate chatbot-style prompt for training."""
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}

        data_entry = samples[ix]

        if isinstance(data_entry, dict):
            user_msg = {"role": "user", "content": data_entry.get("context", data_entry.get("instruction", ""))}
            assistant_msg = {"role": "assistant", "content": data_entry.get("response", data_entry.get("output", ""))}
        else:
            print(f" Warning: Unexpected data format at index {ix}: {type(data_entry)}")
            user_msg = {"role": "user", "content": ""}
            assistant_msg = {"role": "assistant", "content": ""}

        return tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg, assistant_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_vllm_model(self, model_id):
        """Load a vLLM model without Fishfarm dependencies."""
        print(f"Initializing vLLM model: {model_id}")
        model = vllm.LLM(
            model_id,
            max_model_len=1024,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="bfloat16",
            download_dir=get_download_dir(),
        )
        print("Model loaded successfully")
        return model

    def log_metrics(self, step, loss, accuracy):
        """Log training metrics to WandB."""
        wandb.log({"step": step, "loss": loss, "accuracy": accuracy})
        print(f"Logged metrics -> Step: {step}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

class SimpleEvaluator:
    """Basic evaluator class to wrap dataset splits and provide an evaluate() method."""
    def __init__(self, dataset):
        self.samples = dataset  # 🔥 Change 'dataset' to 'samples' for compatibility

    def evaluate(self, model, sample_ids=None):
        """Mock evaluation function—replace with real logic if needed."""
        return {"accuracy": 1.0}  # Dummy return for now

