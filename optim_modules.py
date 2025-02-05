import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logging_utils import get_mean_std_max_min_dict
from utils import (backward, eval_model, forward, load_base_params,
                   load_hf_params_to_vllm)


class OptimizationAlgorithm(abc.ABC):
    def __init__(self, **kwargs):
        nn.Module.__init__(self=self)

    @abc.abstractmethod
    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        original_model_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        raise NotADirectoryError

    @abc.abstractmethod
    def update(self, policy):
        raise NotImplementedError

    def log_optim(self, metrics_to_log):
        pass


class Reinforce(OptimizationAlgorithm, nn.Module):
    def __init__(
        self, policy, gpu, max_grad_norm, lr, rw_norm, rw_clip, kl_ref_coeff, **kwargs
    ):
        nn.Module.__init__(self=self)
        self.gpu = gpu
        self.kl_ref_coeff = kl_ref_coeff
        self.use_kl_loss = kl_ref_coeff > 0.0
        self.max_grad_norm = float(max_grad_norm)
        self.lr = lr
        self.rw_norm = rw_norm
        self.rw_clip = rw_clip
        self.optimizer = torch.optim.Adam(policy.trainable_params, lr=lr)

    def compute_ref_logprobs(
        self,
        model,
        tokenizer,
        prompts,
        res,
    ):
        ref_log_probs_list = []
        print("Computing reference log probs...")
        for j, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.gpu)
            prompt_length = input_ids.shape[-1]
            output_ids = tokenizer(
                prompt + res.get("sample_details", [{}])[j].get("output", ""),
                return_tensors="pt",
            ).input_ids.to(self.gpu)
            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            ref_log_probs_list.append(log_probs.detach().cpu())
        return ref_log_probs_list

    def get_rewards(self, task_loader, res):
        rw_norm = self.rw_norm
        rw_clip = self.rw_clip
        rewards = task_loader.get_rewards(res=res)

        if rw_norm:
            rewards = np.array(rewards)
            mean_rw = np.mean(rewards)
            std_rw = np.clip(np.std(rewards), a_min=1e-7, a_max=None)
            rewards = (rewards - mean_rw) / std_rw
        if rw_clip is not None:
            if rw_clip > 0:
                rewards = np.array(rewards)
                rewards = np.clip(rewards, a_min=-rw_clip, a_max=rw_clip)
        return rewards

    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        original_model_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        use_kl_loss = self.use_kl_loss
        kl_ref_coeff = self.kl_ref_coeff

        gpu = self.gpu

        prompts = [
            task_loader.get_prompt(
                tokenizer,
                train_data,
                i,
                model_id=model_id,
            )
            for i in batch_ix
        ]

        clipped_batch_size = len(prompts)

        learnable_params = policy.get_learnable_params()
        new_params = forward(
            policy, model, base_params, decomposed_params, learnable_params
        )

        print("Loading weights and getting completions with VLLM")
        load_hf_params_to_vllm(new_params, vllm_model)
        res = eval_model(vllm_model, train_eval, batch_ix)
        rewards = self.get_rewards(task_loader=task_loader, res=res)

        rw_stats = get_mean_std_max_min_dict(array=rewards, prefix="rewards")
        metrics_to_log.update(**rw_stats)

        if use_kl_loss:
            with torch.no_grad():
                load_base_params(model=model, base_params=original_model_params)
                ref_log_probs_list = self.compute_ref_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    res=res,
                )
                new_params = forward(
                    policy, model, base_params, decomposed_params, learnable_params
                )

        print("Computing the policy gradient...")
        for j, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu)
            prompt_length = input_ids.shape[-1]

            # 🔥 Check if 'sample_details' is available and within bounds before using it
            if j < len(res.get("sample_details", [])):  
                sample_output = res["sample_details"][j].get("output", "")
            else:
                sample_output = ""  # 🔥 Default to empty if out of range

            prompt += sample_output

            output_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(gpu)
            generated_ids = output_ids[:, prompt_length:]

            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                2, generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            log_likelihood = selected_log_probs.sum(axis=-1)

            pg = -log_likelihood * rewards[j]
            loss = pg

            if use_kl_loss:
                ref_log_probs = ref_log_probs_list[j].to(gpu)
                kl_div = F.kl_div(
                    input=log_probs,
                    target=ref_log_probs,
                    log_target=True,
                    reduction="sum",
                )
                loss = loss + kl_ref_coeff * kl_div
            scaled_loss = loss / clipped_batch_size
            scaled_loss.backward()
            log_dict = {
                "pg": pg.item(),
                "loss": loss.item(),
            }
            if use_kl_loss:
                log_dict["kl_div"] = kl_div.item()
            metrics_to_log.update(**log_dict)
        backward(policy, model, base_params, decomposed_params, learnable_params)

    def update(self, policy):
        max_grad_norm = self.max_grad_norm
        torch.nn.utils.clip_grad_norm_(policy.trainable_params, max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def log_optim(self, metrics_to_log):
        metrics_dict = metrics_to_log.get()
        pg = metrics_dict["pg"]
        print(f"PG={pg}")
        if self.use_kl_loss:
            kl_div = metrics_dict["kl_div"]
            print(f"kl_div={kl_div}")


class RandomShooting(OptimizationAlgorithm, nn.Module):
    def __init__(
        self,
        policy,
        gpu,
        pop_size,
        min_trainable_param,
        max_trainable_param,
        optim_ema=0,
        re_eval_best=True,
        use_loglikelihood_for_ties=False,
        **kwargs,
    ):

        nn.Module.__init__(self=self)
        self.gpu = gpu
        trainable_params = policy.trainable_params
        self.pop_size = pop_size
        self.min_trainable_param = min_trainable_param
        self.max_trainable_param = max_trainable_param
        self.range_trainable_param = max_trainable_param - min_trainable_param
        assert optim_ema >= 0 and optim_ema < 1
        self.optim_ema = optim_ema
        self.re_eval_best = re_eval_best
        self.use_loglikelihood_for_ties = use_loglikelihood_for_ties

        self.trainable_params_shapes = [p.shape for p in trainable_params]
        self.trainable_params_nums = [torch.numel(p) for p in trainable_params]
        self.trainable_params_dtype = trainable_params[0].dtype
        self.total_trainable_params = sum(self.trainable_params_nums)
        self.best_idx = 0

        initial_values = (
            torch.rand(size=[pop_size, self.total_trainable_params])
            * self.range_trainable_param
        ) + self.min_trainable_param
        init_values_flat = [
            torch.flatten(torch.detach_copy(p.data)) for p in trainable_params
        ]
        init_soln = torch.concat(init_values_flat, dim=0)
        if self.re_eval_best:
            if initial_values.shape[1] == init_soln.shape[0]:  
                initial_values[0] = torch.clone(init_soln)  # Only assign if dimensions match
            else:
                print(f"Shape mismatch: {initial_values.shape} vs {init_soln.shape}")  # Debugging output

        self.pop_params = nn.Parameter(
            initial_values,
            requires_grad=False,
        ).cpu()
        self.best_soln = nn.Parameter(init_soln, requires_grad=False).cpu()

    def compute_logprobs(
        self,
        model,
        tokenizer,
        prompts,
        generated_outputs,
    ):
        selected_log_probs_list = []
        for j, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.gpu)
            prompt_length = input_ids.shape[-1]
            output_ids = tokenizer(
                prompt + generated_outputs[j],
                return_tensors="pt",
            ).input_ids.to(self.gpu)
            generated_ids = output_ids[:, prompt_length:]

            outputs = model(output_ids)
            logits = outputs.logits[:, prompt_length - 1 : -1]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                2, generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            selected_log_probs_list.append(selected_log_probs.detach().cpu())
        return selected_log_probs_list

    @torch.no_grad
    def sample_new_params(
        self,
    ):
        pop_values = (
            torch.rand(size=[self.pop_size, self.total_trainable_params])
            * self.range_trainable_param
        ) + self.min_trainable_param
        if self.re_eval_best:
            pop_values[0] = torch.detach_copy(self.best_soln)

        self.pop_params.data.copy_(pop_values)

    def split_and_convert(self, flat_params):
        split_flat_params = torch.split_with_sizes(
            flat_params, split_sizes=self.trainable_params_nums
        )
        split_params = [
            torch.reshape(p, shape=s).to(dtype=self.trainable_params_dtype).to(self.gpu)
            for p, s in zip(split_flat_params, self.trainable_params_shapes)
        ]
        return split_params

    def get_params_for_pop_member(self, pop_idx):
        return self.split_and_convert(self.pop_params[pop_idx])

    @torch.no_grad
    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        self.sample_new_params()
        perf_per_pop = []
        avg_log_likelihoods_per_pop = []
        for pop_idx in range(self.pop_size):
            pop_idx_params = self.split_and_convert(
                flat_params=self.pop_params[pop_idx]
            )
            policy.set_trainable_params_values(new_values=pop_idx_params)
            learnable_params = policy.get_learnable_params()
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )

            print("Loading weights and getting completions with VLLM")
            load_hf_params_to_vllm(new_params, vllm_model.llm)
            res = eval_model(vllm_model, train_eval, batch_ix)
            if self.use_loglikelihood_for_ties:
                print("Storing log likelihhods")
                rewards = task_loader.get_rewards(res=res)
                correct = [int(r > 0) for r in rewards]
                correct_batch_ix = [i for i, c in zip(batch_ix, correct) if c]
                if len(correct_batch_ix) > 0:
                    avg_log_likelihoods = []
                    correct_prompts = [
                        task_loader.get_prompt(
                            tokenizer,
                            train_data,
                            i,
                            model_id=model_id,
                        )
                        for i in correct_batch_ix
                    ]
                    correct_outputs = []
                    for j, c in enumerate(correct):
                        if c:
                            if j < len(res.get("sample_details", [])):
                                correct_outputs.append(res["sample_details"][j].get("output", ""))
                            else:
                                correct_outputs.append("")  # 🔥 Default to empty if out of range

                    selected_log_probs_list = self.compute_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=correct_prompts,
                        generated_outputs=correct_outputs,
                    )
                    for selected_log_probs in selected_log_probs_list:
                        avg_log_likelihood = selected_log_probs.mean(axis=-1)
                        avg_log_likelihoods.append(avg_log_likelihood.item())
                    avg_log_likelihoods_per_pop.append(np.mean(avg_log_likelihoods))
                else:
                    avg_log_likelihoods_per_pop.append(0.0)

            perf = res.aggregate_metrics[task_loader.target_metric_train]
            perf_per_pop.append(perf)

        perf_stats = get_mean_std_max_min_dict(array=perf_per_pop, prefix="pop_perf")
        metrics_to_log.update(**perf_stats)

        if self.use_loglikelihood_for_ties:
            perf_per_pop_array = np.array(perf_per_pop)
            loglikelihood_array = np.array(avg_log_likelihoods_per_pop)
            max_perf = perf_per_pop_array == np.max(perf_per_pop_array)
            max_perf_idxs = np.flatnonzero(max_perf)
            max_perf_logprobs = loglikelihood_array[max_perf_idxs]
            print("SC CHECK")
            print(perf_per_pop)
            print(loglikelihood_array)
            print(max_perf_idxs)
            best_logprob_idx = np.argmax(max_perf_logprobs)
            best_member_idx = max_perf_idxs[best_logprob_idx]
            print(best_logprob_idx)
            print(best_member_idx)
            logprobs_stats = get_mean_std_max_min_dict(
                array=max_perf_logprobs, prefix="logprobs_correct"
            )
            metrics_to_log.update(**logprobs_stats)
        else:
            best_member_idx = np.argmax(perf_per_pop)
        self.best_idx = best_member_idx
        best_params = self.pop_params[best_member_idx].cpu()
        self.best_soln.data.copy_(
            best_params * (1 - self.optim_ema) + self.optim_ema * self.best_soln.cpu()
        )

    def update(self, policy):
        policy.set_trainable_params_values(
            new_values=self.split_and_convert(self.best_soln)
        )

    def log_optim(self, metrics_to_log):
        pass


class CEM(RandomShooting):
    def __init__(
        self,
        policy,
        gpu,
        elite_ratio,
        pop_size,
        min_trainable_param,
        max_trainable_param,
        optim_ema=0,
        re_eval_best=True,
        use_loglikelihood_for_ties=False,
        **kwargs,
    ):

        RandomShooting.__init__(
            self=self,
            policy=policy,
            gpu=gpu,
            pop_size=pop_size,
            min_trainable_param=min_trainable_param,
            max_trainable_param=max_trainable_param,
            optim_ema=optim_ema,
            re_eval_best=re_eval_best,
            use_loglikelihood_for_ties=use_loglikelihood_for_ties,
            **kwargs,
        )

        self.elite_ratio = elite_ratio
        self.num_elites = int(elite_ratio * pop_size)
        self.dist_mean = nn.Parameter(
            torch.detach_copy(self.best_soln), requires_grad=False
        ).cpu()
        init_stdev = (
            torch.ones([self.total_trainable_params]) * self.range_trainable_param / 2
        )
        self.dist_std = nn.Parameter(init_stdev, requires_grad=False).cpu()

    @torch.no_grad
    def sample_new_params(
        self,
    ):
        pop_values = (
            torch.randn(size=[self.pop_size, self.total_trainable_params])
            * self.dist_std
        ) + self.dist_mean
        pop_values = torch.clamp(
            pop_values,
            min=self.min_trainable_param,
            max=self.max_trainable_param,
        )
        if self.re_eval_best:
            pop_values[0] = torch.detach_copy(self.best_soln)

        self.pop_params.data.copy_(pop_values)

    @torch.no_grad
    def step_optimization(
        self,
        model_id,
        model,
        tokenizer,
        policy,
        task_loader,
        batch_ix,
        train_data,
        train_eval,
        base_params,
        decomposed_params,
        metrics_to_log,
        vllm_model=None,
        **kwargs,
    ):
        self.sample_new_params()
        perf_per_pop = []
        avg_log_likelihoods_per_pop = []
        for pop_idx in range(self.pop_size):
            pop_idx_params = self.split_and_convert(
                flat_params=self.pop_params[pop_idx]
            )
            policy.set_trainable_params_values(new_values=pop_idx_params)
            learnable_params = policy.get_learnable_params()
            new_params = forward(
                policy, model, base_params, decomposed_params, learnable_params
            )

            print("Loading weights and getting completions with VLLM")
            load_hf_params_to_vllm(new_params, vllm_model.llm)
            res = eval_model(vllm_model, train_eval, batch_ix)
            if self.use_loglikelihood_for_ties:
                print("Storing log likelihhods")
                rewards = task_loader.get_rewards(res=res)
                correct = [int(r > 0) for r in rewards]
                correct_batch_ix = [i for i, c in zip(batch_ix, correct) if c]
                if len(correct_batch_ix) > 0:
                    avg_log_likelihoods = []
                    correct_prompts = [
                        task_loader.get_prompt(
                            tokenizer,
                            train_data,
                            i,
                            model_id=model_id,
                        )
                        for i in correct_batch_ix
                    ]
                    correct_outputs = []
                    for j, c in enumerate(correct):
                        if c:
                            if j < len(res.get("sample_details", [])):
                                correct_outputs.append(res["sample_details"][j].get("output", ""))
                            else:
                                correct_outputs.append("")  # 🔥 Default to empty if out of range

                    print("lalala, I am hitting the selected_log_probs!")
                    selected_log_probs_list = self.compute_logprobs(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=correct_prompts,
                        generated_outputs=correct_outputs,
                    )
                    for selected_log_probs in selected_log_probs_list:
                        avg_log_likelihood = selected_log_probs.mean(axis=-1)
                        avg_log_likelihoods.append(avg_log_likelihood.item())
                    avg_log_likelihoods_per_pop.append(np.mean(avg_log_likelihoods))
                else:
                    avg_log_likelihoods_per_pop.append(0.0)

            perf = res.aggregate_metrics[task_loader.target_metric_train]
            perf_per_pop.append(perf)

        perf_stats = get_mean_std_max_min_dict(array=perf_per_pop, prefix="pop_perf")
        metrics_to_log.update(**perf_stats)

        if self.use_loglikelihood_for_ties:
            perf_per_pop_array = np.array(perf_per_pop)
            loglikelihood_array = np.array(avg_log_likelihoods_per_pop)
            max_perf = perf_per_pop_array == np.max(perf_per_pop_array)
            max_perf_idxs = np.flatnonzero(max_perf)
            max_perf_logprobs = loglikelihood_array[max_perf_idxs]
            best_logprob_idx = np.argmax(max_perf_logprobs)
            best_member_idx = max_perf_idxs[best_logprob_idx]
            logprobs_stats = get_mean_std_max_min_dict(
                array=max_perf_logprobs, prefix="logprobs_correct"
            )
            metrics_to_log.update(**logprobs_stats)
        else:
            best_member_idx = np.argmax(perf_per_pop)
        elite_idxs = np.argpartition(perf_per_pop, -self.num_elites)[-self.num_elites :]

        elite_params = self.pop_params[elite_idxs]
        elite_mean = torch.mean(elite_params, dim=0)
        elite_std = torch.std(elite_params, dim=0)
        self.best_idx = best_member_idx
        best_params = self.pop_params[best_member_idx].cpu()
        self.best_soln.data.copy_(best_params)
        self.dist_mean.copy_(
            elite_mean.cpu() * (1 - self.optim_ema)
            + self.optim_ema * self.dist_mean.cpu()
        )
        self.dist_std.copy_(
            elite_std.cpu() * (1 - self.optim_ema)
            + self.optim_ema * self.dist_std.cpu()
        )

        cem_mean_stats = get_mean_std_max_min_dict(
            array=self.dist_mean.detach().cpu().numpy(),
            prefix="cem_mean",
        )
        metrics_to_log.update(**cem_mean_stats)

        cem_std_stats = get_mean_std_max_min_dict(
            array=self.dist_std.detach().cpu().numpy(),
            prefix="cem_std",
        )
        metrics_to_log.update(**cem_std_stats)
