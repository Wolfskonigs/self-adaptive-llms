import re
from copy import deepcopy
from typing import Dict, Optional

import fishfarm
import torch
import torch.utils
import vllm


import torch

def load_hf_params_to_vllm(param: Dict, llm: vllm.LLM) -> None:
    """Load weights from HF transformer model to vLLM model without autograd conflicts."""

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    num_layers = model.config.num_hidden_layers

    with torch.no_grad():  # Disable gradient tracking
        # Load embeddings layer weights.
        model_param = model.get_parameter("model.embed_tokens.weight")
        model_param.copy_(
            param["model.embed_tokens.weight"][: model_param.shape[0]]
            .to(model_param.dtype)
            .to(model_param.device)
        )

        model_param = model.get_parameter("lm_head.weight")
        model_param.copy_(
            param["lm_head.weight"][: model_param.shape[0]]
            .to(model_param.dtype)
            .to(model_param.device)
        )

        # Load the final layernorm weights.
        model_param = model.get_parameter("model.norm.weight")
        model_param.copy_(
            param["model.norm.weight"].to(model_param.dtype).to(model_param.device)
        )

        for i in range(num_layers):
            # Load qkv_proj weights.
            model_param = model.get_parameter(f"model.layers.{i}.self_attn.qkv_proj.weight")
            model_param.copy_(
                torch.cat(
                    [
                        param[f"model.layers.{i}.self_attn.q_proj.weight"],
                        param[f"model.layers.{i}.self_attn.k_proj.weight"],
                        param[f"model.layers.{i}.self_attn.v_proj.weight"],
                    ],
                    dim=0,
                )
                .to(model_param.dtype)
                .to(model_param.device)
            )

            # Load gate_up_proj weights.
            model_param = model.get_parameter(f"model.layers.{i}.mlp.gate_up_proj.weight")
            model_param.copy_(
                torch.cat(
                    [
                        param[f"model.layers.{i}.mlp.gate_proj.weight"],
                        param[f"model.layers.{i}.mlp.up_proj.weight"],
                    ],
                    dim=0,
                )
                .to(model_param.dtype)
                .to(model_param.device)
            )

            # Load o_proj and down_proj weights.
            model_param = model.get_parameter(f"model.layers.{i}.self_attn.o_proj.weight")
            model_param.copy_(
                param[f"model.layers.{i}.self_attn.o_proj.weight"]
                .to(model_param.dtype)
                .to(model_param.device)
            )

            model_param = model.get_parameter(f"model.layers.{i}.mlp.down_proj.weight")
            model_param.copy_(
                param[f"model.layers.{i}.mlp.down_proj.weight"]
                .to(model_param.dtype)
                .to(model_param.device)
            )

            # Load layer_norm weights.
            model_param = model.get_parameter(f"model.layers.{i}.input_layernorm.weight")
            model_param.copy_(
                param[f"model.layers.{i}.input_layernorm.weight"]
                .to(model_param.dtype)
                .to(model_param.device)
            )

            model_param = model.get_parameter(
                f"model.layers.{i}.post_attention_layernorm.weight"
            )
            model_param.copy_(
                param[f"model.layers.{i}.post_attention_layernorm.weight"]
                .to(model_param.dtype)
                .to(model_param.device)
            )


def eval_model(vllm_model, evaluator, ix=None):
    result = evaluator.evaluate(vllm_model, sample_ids=ix)
    return result


def compose_new_params(
    policy,
    param_name,
    decomposed_params,
    learnable_params,
):
    """Compose new parameters from decomposed parameters."""
    mm = policy.get_mask(learnable_params[param_name])
    return (
        decomposed_params[f"{param_name}.U"]
        @ torch.diag_embed(decomposed_params[f"{param_name}.S"] * mm)
        @ decomposed_params[f"{param_name}.V"].T
    ) * (
        decomposed_params[f"{param_name}.S"].sum()
        / (decomposed_params[f"{param_name}.S"] * mm).sum()
    )


@torch.no_grad()
def forward(policy, model, base_params, decomposed_params, learnable_params):
    """Forward pass."""
    new_params = {}
    for k in base_params:
        if "mlp" in k:
            new_params[k] = compose_new_params(
                policy, k, decomposed_params, learnable_params
            )
            model.get_parameter(k).copy_(new_params[k])
        else:
            new_params[k] = base_params[k]
    return new_params


@torch.no_grad()
def load_base_params(
    model,
    base_params,
):
    for k in base_params:
        if "mlp" in k:
            model.get_parameter(k).copy_(base_params[k].cuda())


def backward(
    policy,
    model,
    base_params,
    decomposed_params,
    learnable_params,
):
    """Backward pass."""
    keys_to_backprop = [k for k in base_params if "mlp" in k]
    last_key = keys_to_backprop[-1]
    for k in keys_to_backprop[:-1]:
        compose_new_params(policy, k, decomposed_params, learnable_params).backward(
            model.get_parameter(k).grad, retain_graph=True
        )
    # release graph
    compose_new_params(policy, last_key, decomposed_params, learnable_params).backward(
        model.get_parameter(last_key).grad, retain_graph=False
    )


def classify_samples(vllm_model, test_eval):
    """Classify samples."""

    CLASSIFICATION_PROMPT = """
    # Analyze the given question and classify it into one of four categories: 'code', 'math', 'reasoning' or 'other'. Follow these guidelines:

    1. Code: Questions asking for programming solutions, functions, algorithms. Often includes specific programming terms, language syntax, or data structures.
    2. Math: Questions involving mathematical calculations, formulas, statistics. Often includes numbers, equations, or mathematical operations.
    3. Reasoning: Questions requiring logical thinking, application of scientific knowledge, or critical analysis of information. Often presents statements that need evaluation based on general understanding. 
    4. Other: Questions not clearly fit into above categories.

    Instructions:
    - Consider the primary focus, skills, and knowledge required to answer the question.
    - If a question spans multiple categories, choose the most dominant one.
    - Provide your final classification within \\boxed{} notation. Example: \\boxed{reasoning}

    Format your response as follows:
    Classification: \\boxed{category}
    """

    def extract_classification(text: str) -> Optional[str]:
        """
        Extract the classification from the model's output using regex.
        """
        match = re.search(r"\\boxed{([^}]*)}", text)
        return match.group(1) if match else None

    # Identify the key in the samples that contains the problem text
    problem_key = None
    for key in ("problem", "question", "instruction"):
        if (
            hasattr(test_eval.samples[0], key)
            and getattr(test_eval.samples[0], key) is not None
        ):
            problem_key = key
            break
    assert problem_key is not None, "Could not find problem text in the samples"

    # Prepare classification requests
    classification_requests = [
        fishfarm.models.GenerationRequest(
            messages=[
                fishfarm.Message("system", CLASSIFICATION_PROMPT),
                fishfarm.Message("user", getattr(sample, problem_key)),
            ]
        )
        for sample in test_eval.samples
    ]

    # Generate classifications using the model
    model_outputs = vllm_model.generate(classification_requests)

    # Process results and update samples
    classified_samples = []
    for sample, result in zip(test_eval.samples, model_outputs):
        prediction = extract_classification(result.generation)
        if prediction not in ["code", "math", "reasoning"]:
            prediction = "other"
        sample.expert_label = prediction
        classified_samples.append(sample)

    return classified_samples


def eval_model_experts_prompt_based(
    vllm_model,
    evaluator,
    experts_path_dict,
    policy,
    model,
    base_params,
    decomposed_params,
    task_metric,
):
    """Evaluate the model using expert models and prompt-based classification."""
    results_by_expert: Dict[str, Dict] = {}

    # Classify all test samples
    classified_samples = classify_samples(vllm_model, evaluator)

    # Evaluate samples for each expert model
    for expert_label, expert_model_path in experts_path_dict.items():
        # Filter samples for current expert
        expert_samples = [
            sample
            for sample in classified_samples
            if sample.expert_label == expert_label
        ]
        if not expert_samples:
            continue

        # Update test evaluation with filtered samples
        evaluator.samples = expert_samples

        # Load and apply expert model parameters if available
        if expert_model_path:
            policy.load_state_dict(torch.load(expert_model_path))
            expert_params = policy.get_learnable_params()
            updated_params = forward(
                policy=policy,
                model=model,
                base_params=base_params,
                decomposed_params=decomposed_params,
                learnable_params=expert_params,
            )
            load_hf_params_to_vllm(updated_params, vllm_model.llm)

        # Evaluate current expert model
        evaluation_results = eval_model(vllm_model, evaluator)

        # Store results for current expert
        results_by_expert[expert_label] = {
            "num_samples": len(expert_samples),
            "test_acc": evaluation_results.aggregate_metrics[task_metric],
        }

    # Compute the overall accuracy.
    data_dict = deepcopy(results_by_expert)
    data_dict["final_test_acc"] = 0.0
    for label in results_by_expert.keys():
        data_dict["final_test_acc"] += (
            results_by_expert[label]["test_acc"]
            * results_by_expert[label]["num_samples"]
        )
    data_dict["final_test_acc"] /= len(classified_samples)

    return data_dict
