import gc
import json
import os
from datetime import datetime
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from base_model import BaseModel
from logging_utils import Metrics, get_mean_std_max_min_dict
from optim_modules import OptimizationAlgorithm
from policy import Policy
from tasks import Task
from utils import (eval_model, eval_model_experts_prompt_based, forward,
                   load_hf_params_to_vllm)


def wandb_init(cfg, run_name: str, group_name: str, log_dir: str):
    import wandb

    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=False,
    )
    config_dict["log_dir"] = log_dir
    config_dict["wandb_run_name"] = run_name
    config_dict["wandb_group_name"] = group_name

    # wandb has a 128-size character limit on the group name
    wandb.init(
        project=cfg.wandb_project,
        group=group_name[:127],
        name=run_name[:127],
        config=config_dict,
    )
    return wandb


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg):
    """Main function."""

    num_iters = cfg.num_iters
    test_interval = cfg.test_interval

    batch_size = cfg.batch_size
    seed = cfg.seed
    policy_name = cfg.policy_name
    test_only = cfg.test_only
    save_legacy_params = cfg.save_legacy_params
    exp_name = cfg.exp_name
    run_name = cfg.run_name

    task_name = cfg.task_name

    load_ckpt = cfg.load_ckpt
    use_lora = cfg.use_lora
    prompt_based_eval = cfg.prompt_based_eval
    experts_path_dict = cfg.experts_path_dict

    resuming_from_ckpt = False
    if load_ckpt is not None:
        if load_ckpt == "scratch" or load_ckpt == "base":
            resuming_from_ckpt = False
        else:
            resuming_from_ckpt = True

    # Create task
    task_loader: Task = hydra.utils.instantiate(cfg.task_loader)

    base_model: BaseModel = hydra.utils.instantiate(cfg.base_model)

    model_id = base_model.get_model_id()
    decomposed_param_file = base_model.get_param_file(param_folder_path="")

    extract_svd = cfg.extract_svd or (not os.path.exists(decomposed_param_file))

    has_training_split = task_loader.has_training_split
    has_transfer_split = task_loader.has_transfer_split

    if not has_training_split:
        assert test_only, "Cannot train on a task with no training split"

    if exp_name is None:
        exp_name = "temp"

    metrics_to_log = Metrics()

    # Create log dir.
    if run_name is None:
        now = datetime.now()
        run_name = now.strftime("%Y%m%d-%H%M%S")
    if test_only and (not resuming_from_ckpt):
        log_dir = f"{cfg.out_dir}/{task_name}/{cfg.base_model_name}_base"
        group_name = cfg.base_model_name
    else:
        log_dir = f"{cfg.out_dir}/{task_name}/{policy_name}/{exp_name}/{run_name}"
        group_name = cfg.wandb_group_name
    os.makedirs(log_dir, exist_ok=True)

    vllm_model = task_loader.get_vllm_model(model_id=model_id)

    # Properly handle the evaluator return type
    data_splits = task_loader.get_evaluator()

    # Ensure it's a tuple for predictable unpacking
    if isinstance(data_splits, tuple):  
        train_eval, test_evals = data_splits  
    elif isinstance(data_splits, list):  
        train_eval = SimpleEvaluator(data_splits[0]) if len(data_splits) > 0 else None
        test_eval = SimpleEvaluator(data_splits[1]) if len(data_splits) > 1 else None
    else:
        train_eval = SimpleEvaluator(data_splits)  # Wrap a single dataset in evaluator
        test_eval = None

    # Ensure test_evals is properly unpacked
    if isinstance(test_evals, (list, tuple)) and len(test_evals) > 0:
        test_eval = test_evals[0]  # If it’s a list/tuple, grab the first element
    else:
        test_eval = test_evals  # If it’s already an object, just assign it directly

    # Final sanity check
    if train_eval is None or not hasattr(train_eval, "evaluate"):
        raise TypeError(f"train_eval is invalid: {type(train_eval)}, expected an evaluator object")

    train_data, train_ix, valid_ix = task_loader.get_train_data()
    gpu = torch.device("cuda:0")
    np_random = np.random.RandomState(seed)

    # cpu + float32 for initial SVD decomposition
    if extract_svd:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32
        )
    else:
        # Load model and tokenizer.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda:0", torch_dtype=torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_params = model.state_dict()

    original_model_params = {
        k: v.clone().detach().cpu() for k, v in base_params.items() if "mlp" in k
    }

    # Load decomposed parameters.
    if not os.path.exists(decomposed_param_file):
        print("Decomposed params not found. Decomposing...")
        decomposed_params = {}
        for k, v in base_params.items():
            if "norm" not in k:
                print(k)
                U, S, V = torch.svd(v)
                decomposed_params[f"{k}.U"] = U
                decomposed_params[f"{k}.S"] = S
                decomposed_params[f"{k}.V"] = V
        torch.save(decomposed_params, decomposed_param_file)
        print("successfully decomposed model - returning")
        return
    elif extract_svd:
        print(f"ERROR: SVD file already exists at {decomposed_param_file}")
    else:
        print("Decomposed params found. Loading...")
        assert not extract_svd
        decomposed_params = torch.load(decomposed_param_file)
    for k, v in decomposed_params.items():
        decomposed_params[k] = v.to(torch.bfloat16).to(gpu)

    if cfg.wandb_log:
        wandb = wandb_init(
            cfg=cfg, group_name=group_name, run_name=run_name, log_dir=log_dir
        )

    policy: Policy = hydra.utils.instantiate(
        cfg.shakeoff_policy,
        base_params=base_params,
        decomposed_params=decomposed_params,
        gpu=gpu,
    )

    optimization_algorithm: OptimizationAlgorithm = hydra.utils.instantiate(
        cfg.optimization_algorithm,
        policy=policy,
        gpu=gpu,
    )

    if resuming_from_ckpt and os.path.exists(load_ckpt):
        print(f"Starting from checkpoint at: {load_ckpt}")
        # load the lora weight
        if use_lora:
            assert os.path.isdir(load_ckpt), "ckpt for lora must be dir to lora adapter"
            from peft import PeftModel

            lora_model = PeftModel.from_pretrained(model, load_ckpt)
            merged_model = lora_model.merge_and_unload()
            new_params = merged_model.state_dict()
        # load svd expert
        elif "learnable_params" in load_ckpt:
            learnable_params = torch.load(load_ckpt)
            for k, v in learnable_params.items():
                learnable_params[k] = v.to(gpu)
            assert test_only
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )
        else:
            state_dict = torch.load(load_ckpt, weights_only=True)
            policy.load_state_dict(state_dict=state_dict)
            if test_only:
                learnable_params = policy.get_learnable_params()
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )
        load_hf_params_to_vllm(new_params, vllm_model.llm)
    else:
        print(f"Starting from the base model as load_ckpt=={load_ckpt}")

    model.eval()

    # Prompt-based and cls dispatcher evaluation.
    if test_only and prompt_based_eval:
        test_data_dict = eval_model_experts_prompt_based(
            vllm_model,
            test_eval,
            experts_path_dict,
            policy,
            model,
            base_params,
            decomposed_params,
            task_loader.target_metric_test,
        )
        test_data_dict["type"] = "test"
        # Log the results.
        if cfg.wandb_log:
            wandb.log(test_data_dict)
        with open(f"{log_dir}/eval_results.json", "w") as f:
            json.dump(test_data_dict, f, indent=4)
        print(f"Test evaluation results: {test_data_dict}")

        # Eval the transfer set if available
        if has_transfer_split:
            transfer_data_dict = eval_model_experts_prompt_based(
                vllm_model,
                transfer_eval,
                experts_path_dict,
                policy,
                model,
                base_params,
                decomposed_params,
                task_loader.target_metric_transfer,
            )
            transfer_data_dict["type"] = "transfer"
            # Log the results.
            if cfg.wandb_log:
                wandb.log(transfer_data_dict)
            with open(f"{log_dir}/eval_results.json", "w") as f:
                json.dump(transfer_data_dict, f, indent=4)
            print(f"Transfer evaluation results: {transfer_data_dict}")

        return

    # Non-adaptive evaluation on train, val, test set.
    if test_only and not prompt_based_eval:
        data_dict = {}
        details_dict = {}
        if has_training_split:
            train_res = eval_model(vllm_model, train_eval, train_ix)
            valid_res = eval_model(vllm_model, train_eval, valid_ix)
            data_dict["train_acc"] = train_res.aggregate_metrics[
                task_loader.target_metric_train
            ]
            data_dict["valid_acc"] = valid_res.aggregate_metrics[
                task_loader.target_metric_valid
            ]
            details_dict["train"] = train_res.sample_details
            details_dict["valid"] = valid_res.sample_details
        test_res = eval_model(vllm_model, test_eval)
        data_dict["test_acc"] = test_res.aggregate_metrics[
            task_loader.target_metric_test
        ]
        details_dict["test"] = test_res.sample_details
        if has_transfer_split:
            transfer_res = eval_model(vllm_model, transfer_eval)
            data_dict["transfer_acc"] = transfer_res.aggregate_metrics[
                task_loader.target_metric_transfer
            ]
            details_dict["transfer"] = transfer_res.sample_details
        if cfg.wandb_log:
            wandb.log(data_dict)
        with open(f"{log_dir}/eval_results.json", "w") as f:
            json.dump(data_dict, f, indent=4)
        print(f"Evaluation results: {data_dict}")
        return

    learnable_params = policy.get_learnable_params()
    for k in learnable_params:
        model.get_parameter(k).requires_grad_(True)

    # Training loop.
    if batch_size is None:
        clipped_batch_size = len(list(train_ix))
    else:
        clipped_batch_size = min(batch_size, len(list(train_ix)))
    best_val_acc = 0.0
    test_at_best = 0.0
    transfer_at_best = 0.0
    for i in range(num_iters):

        batch_ix = np_random.choice(train_ix, size=clipped_batch_size, replace=False)

        optimization_algorithm.step_optimization(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            policy=policy,
            task_loader=task_loader,
            batch_ix=batch_ix,
            train_data=train_data,
            train_eval=train_eval,
            base_params=base_params,
            decomposed_params=decomposed_params,
            original_model_params=original_model_params,
            metrics_to_log=metrics_to_log,
            vllm_model=vllm_model,
        )

        with torch.no_grad():
            lists_to_log = {}
            grads = [p.grad for p in policy.trainable_params]
            if grads[0] is not None:
                grad_mean = [g.mean().item() for g in grads]
                grad_mags = [torch.linalg.vector_norm(g).item() for g in grads]
                lists_to_log["grad_mean"] = grad_mean
                lists_to_log["grad_mags"] = grad_mags

                param_mags = [
                    torch.linalg.vector_norm(p).item() for p in policy.trainable_params
                ]
                lists_to_log["policy_param_mag"] = param_mags

            generated_params_list = list(learnable_params.values())

            generated_param_mean = [p.mean().item() for p in generated_params_list]
            generated_param_mags = [
                torch.linalg.vector_norm(p).item() for p in generated_params_list
            ]
            lists_to_log["generated_param_mean"] = generated_param_mean
            lists_to_log["generated_param_mags"] = generated_param_mags

            list_stats = {}
            for k, v in lists_to_log.items():
                list_stats.update(get_mean_std_max_min_dict(array=v, prefix=k))
            metrics_to_log.update(**list_stats)

        optimization_algorithm.update(policy=policy)

        # Make sure old params are deleted and garbage-collected
        gc.collect()
        torch.cuda.empty_cache()
        model.zero_grad()

        # More accurate logging.
        value_mean = list_stats.get("generated_param_mean/mean", None)
        grad_mean_mag = list_stats.get("grad_mags/mean", None)
        print(
            f"Iter {i}: "
            + f"param_mean={value_mean}, "
            + f"grad_mean_mag={grad_mean_mag}"
        )
        optimization_algorithm.log_optim(metrics_to_log=metrics_to_log)

        # Test and save.
        if i % test_interval == 0:
            learnable_params = policy.get_learnable_params()
            forward(policy, model, base_params, decomposed_params, learnable_params)
            load_hf_params_to_vllm(model.state_dict(), vllm_model.llm)

            train_res = eval_model(vllm_model, train_eval, train_ix)
            valid_res = eval_model(vllm_model, train_eval, valid_ix)
            test_res = eval_model(vllm_model, test_eval)
            if has_transfer_split:
                transfer_res = eval_model(vllm_model, transfer_eval)
            if (
                valid_res.aggregate_metrics[task_loader.target_metric_valid]
                > best_val_acc
            ):
                best_val_acc = valid_res.aggregate_metrics[
                    task_loader.target_metric_valid
                ]
                test_at_best = test_res.aggregate_metrics[
                    task_loader.target_metric_test
                ]
                if has_transfer_split:
                    transfer_at_best = transfer_res.aggregate_metrics[
                        task_loader.target_metric_transfer
                    ]
                print("best_val_acc updated")
                path = f"{log_dir}/policy_params.pt"
                torch.save(policy.state_dict(), path)
                if save_legacy_params:
                    torch.save(learnable_params, f"{log_dir}/learnable_params.pt")

            path = f"{log_dir}/policy_params_latest.pt"
            torch.save(policy.state_dict(), path)
            if save_legacy_params:
                torch.save(learnable_params, f"{log_dir}/learnable_params_latest.pt")

            policy.record_state(metrics_to_log=metrics_to_log)
            data_dict = {
                "iter": i,
                "best_val_acc": best_val_acc,
                "test_at_best_val": test_at_best,
                "train_acc": train_res.aggregate_metrics[
                    task_loader.target_metric_train
                ],
                "valid_acc": valid_res.aggregate_metrics[
                    task_loader.target_metric_valid
                ],
                "test_acc": test_res.aggregate_metrics[task_loader.target_metric_test],
                **metrics_to_log.get(),
            }
            if has_transfer_split:
                data_dict["transfer_acc"] = transfer_res.aggregate_metrics[
                    task_loader.target_metric_transfer
                ]
                data_dict["transfer_at_best_val"] = transfer_at_best
            if cfg.wandb_log:
                wandb.log(data_dict)
            with open(f"{log_dir}/reinforce_log.json", "a") as f:
                json_data = json.dumps(data_dict, indent=4)
                f.write(json_data)
                f.write("\n")
            metrics_to_log.reset()


if __name__ == "__main__":
    main()
