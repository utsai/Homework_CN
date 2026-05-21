"""Training script for the 3-context Go/No-Go group coursework.

Created by Cai Chunqiu on May 7, 14:18.
"""

import argparse
import csv
import json
import os
import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from full_rank_rnn import FullRankRNN
from low_rank_rnn import LowRankRNN
from lstm_rnn import LSTMRNN


Rollout = namedtuple("Rollout", ("output", "target", "context", "stimulus"))

class ContextGoNoGoTask:
    """Inline short-context task used by the homework trainer.

    Channels: [stimulus, context_0, context_1, context_2, go_cue].
    The context cue appears only at t=1. The stimulus appears only at t=0.
    All task inputs are off during the delay until the final go cue. The
    network is read out only at the final go-cue timestep.
    """

    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.input_dim = 5
        self.num_contexts = 3
        self.context_t = 1
        self.stimulus_t = 0
        self.go_cue_t = seq_len - 1
        if self.seq_len < 3:
            raise ValueError("seq_len must be at least 3 for context, stimulus, and go cue timesteps.")

    def make_targets(self, contexts, stimuli):
        targets = T.zeros_like(stimuli)
        targets = T.where(contexts == 1, stimuli, targets)
        targets = T.where(contexts == 2, -stimuli, targets)
        return targets.unsqueeze(1)
    def build_inputs(self, contexts, stimuli):
        batch_size = contexts.numel()
        inputs = T.zeros(batch_size, self.seq_len, self.input_dim, device=contexts.device)
        inputs[:, self.context_t, 1:4] = F.one_hot(contexts, self.num_contexts).float()
        inputs[:, self.stimulus_t, 0] = stimuli
        inputs[:, self.go_cue_t, 4] = 1.0
        return inputs

        
# class ContextGoNoGoTask:
#     """Inline 3-context task used by the homework trainer.

#     Channels: [stimulus, context_0, context_1, context_2, go_cue].
#     The context cue is tonic, the stimulus is brief, and the network is read
#     out only at the final go-cue timestep.
#     """

#     def __init__(self, seq_len=10):
#         self.seq_len = seq_len
#         self.input_dim = 5
#         self.num_contexts = 3
#         self.stimulus_t = 0
#         self.go_cue_t = seq_len - 1

#     def make_targets(self, contexts, stimuli):
#         targets = T.zeros_like(stimuli)
#         targets = T.where(contexts == 1, stimuli, targets)
#         targets = T.where(contexts == 2, -stimuli, targets)
#         return targets.unsqueeze(1)

#     def build_inputs(self, contexts, stimuli):
#         batch_size = contexts.numel()
#         inputs = T.zeros(batch_size, self.seq_len, self.input_dim, device=contexts.device)
#         inputs[:, :, 1:4] = F.one_hot(contexts, self.num_contexts).float().unsqueeze(1)
#         inputs[:, self.stimulus_t, 0] = stimuli
#         inputs[:, self.go_cue_t, 4] = 1.0
#         return inputs

    def make_batch(self, batch_size, device):
        contexts = T.randint(0, self.num_contexts, (batch_size,), device=device)
        stimulus_sign = T.randint(0, 2, (batch_size,), device=device)
        stimuli = stimulus_sign.float() * 2.0 - 1.0
        inputs = self.build_inputs(contexts, stimuli)
        targets = self.make_targets(contexts, stimuli)
        return inputs, targets, contexts, stimuli

    def make_balanced_batch(self, repeats_per_case, device):
        cases = [(c, s) for c in range(self.num_contexts) for s in (-1.0, 1.0)]
        contexts = T.tensor(
            [c for c, _ in cases for _ in range(repeats_per_case)],
            dtype=T.long,
            device=device,
        )
        stimuli = T.tensor(
            [s for _, s in cases for _ in range(repeats_per_case)],
            dtype=T.float32,
            device=device,
        )
        inputs = self.build_inputs(contexts, stimuli)
        targets = self.make_targets(contexts, stimuli)
        return inputs, targets, contexts, stimuli

    @staticmethod
    def discretize(outputs, threshold=1.0 / 3.0):
        preds = T.zeros_like(outputs)
        preds = T.where(outputs > threshold, T.ones_like(preds), preds)
        preds = T.where(outputs < -threshold, -T.ones_like(preds), preds)
        return preds

    @staticmethod
    def value_to_index(values):
        return (values.squeeze(1).long() + 1).clamp(0, 2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)


def choose_device(device_arg):
    if device_arg != "auto":
        return T.device(device_arg)
    if T.cuda.is_available():
        return T.device("cuda")
    return T.device("cpu")


class Trainer:
    """Single-stage supervised trainer for the context Go/No-Go task."""

    def __init__(self, args):
        self.args = args
        self.device = choose_device(args.device)
        self.seed = args.seed
        set_seed(self.seed)

        self.task = ContextGoNoGoTask(seq_len=args.seq_len)
        self.repeats_per_case = args.repeats_per_case
        self.batch_size = self.task.num_contexts * 2 * self.repeats_per_case
        self.hidden_size = args.hidden_size
        self.rank = args.rank
        self.alpha = args.alpha
        self.output_size = 1
        self.model_name = args.model

        self.agent = self._build_agent().to(self.device)
        self.optim = T.optim.Adam(self.agent.parameters(), lr=args.lr)
        self.max_grad_norm = args.grad_clip

        self.run_dir = Path(args.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(self.run_dir / ".mplconfig"))

        self._hist_steps = []
        self._hist_loss = []
        self._hist_eval_mse = []
        self._hist_accuracy = []
        self._hist_grad = []
        self._hist_context_acc = {0: [], 1: [], 2: []}

        print(f"[Init] model={self.model_name}, device={self.device}")
        print(
            f"[Init] hidden={self.hidden_size}, rank={self.rank}, alpha={self.alpha}, "
            f"batch={self.batch_size} ({self.repeats_per_case} repeats/case)"
        )
        n_params = sum(p.numel() for p in self.agent.parameters())
        n_train = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        print(f"[Init] Trainable: {n_train:,} / Total: {n_params:,}")

    def _build_agent(self):
        common = {
            "input_size": self.task.input_dim,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }
        if self.model_name == "low_rank":
            return LowRankRNN(rank=self.rank, alpha=self.alpha, **common)
        if self.model_name == "full_rank":
            return FullRankRNN(alpha=self.alpha, **common)
        if self.model_name == "lstm":
            return LSTMRNN(**common)
        raise ValueError(f"Unknown model type: {self.model_name}")

    # -------------------------------------------------------------- #
    #  Run one supervised batch
    # -------------------------------------------------------------- #
    def run_episode(self, episode):
        inputs, targets, contexts, stimuli = self.task.make_balanced_batch(
            self.repeats_per_case, self.device
        )
        full_output = self.agent(inputs)
        final_output = full_output[:, -1, :]
        buffer = [Rollout(final_output, targets, contexts, stimuli)]
        return buffer

    # -------------------------------------------------------------- #
    #  Supervised loss
    # -------------------------------------------------------------- #
    def compute_loss(self, buffer):
        output = T.cat([item.output for item in buffer]).to(self.device)
        target = T.cat([item.target for item in buffer]).to(self.device)
        return F.mse_loss(output, target.float())

    # -------------------------------------------------------------- #
    #  Evaluation helpers
    # -------------------------------------------------------------- #
    @T.no_grad()
    def evaluate(self, repeats_per_case=64):
        self.agent.eval()
        inputs, targets, contexts, stimuli = self.task.make_balanced_batch(
            repeats_per_case, self.device
        )
        outputs = self.agent(inputs)[:, -1, :]
        pred_values = self.task.discretize(outputs)

        target_idx = self.task.value_to_index(targets)
        pred_idx = self.task.value_to_index(pred_values)
        accuracy = (target_idx == pred_idx).float().mean().item()
        mse = F.mse_loss(outputs, targets).item()

        context_accuracy = {}
        for context_id in range(self.task.num_contexts):
            mask = contexts == context_id
            context_accuracy[context_id] = (target_idx[mask] == pred_idx[mask]).float().mean().item()

        confusion = T.zeros(3, 3, dtype=T.long, device=self.device)
        for true_i, pred_i in zip(target_idx, pred_idx):
            confusion[true_i, pred_i] += 1

        rows = []
        for context in range(self.task.num_contexts):
            for stimulus in (-1.0, 1.0):
                mask = (contexts == context) & (stimuli == stimulus)
                mean_output = outputs[mask].mean().item()
                target = self.task.make_targets(
                    T.tensor([context], device=self.device),
                    T.tensor([stimulus], device=self.device),
                ).item()
                pred = self.task.discretize(T.tensor([[mean_output]], device=self.device)).item()
                rows.append(
                    {
                        "context": context,
                        "stimulus": int(stimulus),
                        "target": int(target),
                        "mean_output": mean_output,
                        "prediction": int(pred),
                    }
                )

        self.agent.train()
        return accuracy, mse, context_accuracy, confusion.cpu().numpy(), rows

    # -------------------------------------------------------------- #
    #  Plotting
    # -------------------------------------------------------------- #
    def plot_training_curves(self, save_path=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160)
        axes = axes.flatten()

        axes[0].plot(self._hist_steps, self._hist_loss, color="C0", label="train")
        axes[0].plot(self._hist_steps, self._hist_eval_mse, color="C1", label="balanced eval")
        axes[0].set_title("MSE loss")
        axes[0].set_xlabel("Training step")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self._hist_steps, self._hist_accuracy, color="black", label="all contexts")
        for context_id, values in self._hist_context_acc.items():
            axes[1].plot(self._hist_steps, values, label=f"context {context_id}")
        axes[1].set_title("Learning curve by context")
        axes[1].set_xlabel("Training step")
        axes[1].set_ylabel("Thresholded accuracy")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self._hist_steps, self._hist_grad, color="C2")
        axes[2].set_title("Gradient norm")
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("Norm")
        axes[2].grid(True, alpha=0.3)

        axes[3].axis("off")
        axes[3].text(
            0.02,
            0.95,
            "Task rule\n"
            "Context 0: suppress -> 0\n"
            "Context 1: copy stimulus\n"
            "Context 2: invert stimulus\n\n"
            "Input channels\n"
            "[stimulus, C0, C1, C2, go cue]\n"
            "Context is on all trial; stimulus at t=0; go cue at final step.",
            ha="left",
            va="top",
            fontsize=10,
        )

        fig.suptitle(f"{self.model_name} on 3-context Go/No-Go")
        fig.tight_layout()
        if save_path is None:
            save_path = self.run_dir / f"{self.model_name}_training_curves.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved -> {save_path}")

    def plot_confusion_matrix(self, confusion, save_path=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["-1", "0", "+1"]
        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=160)
        image = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(3), labels=labels)
        ax.set_yticks(range(3), labels=labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("target")
        ax.set_title("Confusion matrix")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(confusion[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        if save_path is None:
            save_path = self.run_dir / f"{self.model_name}_confusion_matrix.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def save_example_rows(self, rows):
        path = self.run_dir / f"{self.model_name}_example_predictions.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["context", "stimulus", "target", "mean_output", "prediction"],
            )
            writer.writeheader()
            writer.writerows(rows)

    # -------------------------------------------------------------- #
    #  Train: single-stage supervised learning
    # -------------------------------------------------------------- #
    def train(self, max_episodes=None, save_interval=None):
        if max_episodes is None:
            max_episodes = self.args.steps
        if save_interval is None:
            save_interval = self.args.save_interval

        best_loss = float("inf")
        progress = tqdm(range(1, max_episodes + 1))

        for episode in progress:
            buffer = self.run_episode(episode)
            loss = self.compute_loss(buffer)

            self.optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optim.step()

            should_log = (
                episode == 1
                or episode % self.args.log_interval == 0
                or episode == max_episodes
            )
            if should_log:
                accuracy, eval_mse, context_accuracy, _, _ = self.evaluate()
                self._hist_steps.append(episode)
                self._hist_loss.append(float(loss.item()))
                self._hist_eval_mse.append(eval_mse)
                self._hist_accuracy.append(accuracy)
                self._hist_grad.append(float(grad_norm))
                for context_id in range(self.task.num_contexts):
                    self._hist_context_acc[context_id].append(context_accuracy[context_id])

                progress.set_description(
                    f"Ep {episode}/{max_episodes} | "
                    f"L:{loss.item():.5f} | Eval:{eval_mse:.5f} | Acc:{accuracy:.3f}"
                )

            if loss.item() < best_loss:
                best_loss = loss.item()

            if save_interval > 0 and episode % save_interval == 0:
                self.save_checkpoint(tag=f"{episode:04d}", episode=episode)

        accuracy, eval_mse, context_accuracy, confusion, rows = self.evaluate(repeats_per_case=128)
        self.save_checkpoint(tag="final", episode=max_episodes, accuracy=accuracy, eval_mse=eval_mse)
        self.plot_training_curves()
        self.plot_confusion_matrix(confusion)
        self.save_example_rows(rows)
        self.save_metrics(accuracy, eval_mse, context_accuracy, confusion)

        print(f"Final balanced accuracy: {accuracy:.3f}")
        print(f"Final context accuracy: {context_accuracy}")
        print(f"Saved outputs to: {self.run_dir}")

    def save_checkpoint(self, tag, episode, accuracy=None, eval_mse=None):
        T.save(
            {
                "model_type": self.model_name,
                "model_state_dict": self.agent.state_dict(),
                "args": vars(self.args),
                "episode": episode,
                "accuracy": accuracy,
                "eval_mse": eval_mse,
            },
            self.run_dir / f"{self.model_name}_{tag}_model.pt",
        )

    def save_metrics(self, accuracy, eval_mse, context_accuracy, confusion):
        metrics = {
            "final_accuracy": accuracy,
            "final_mse": eval_mse,
            "final_context_accuracy": context_accuracy,
            "history": {
                "step": self._hist_steps,
                "loss": self._hist_loss,
                "eval_mse": self._hist_eval_mse,
                "accuracy": self._hist_accuracy,
                "grad_norm": self._hist_grad,
                "context_accuracy": self._hist_context_acc,
            },
            "confusion_matrix": confusion.tolist(),
        }
        with (self.run_dir / f"{self.model_name}_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train RNNs on a 3-context Go/No-Go task.")
    parser.add_argument("--model", choices=["low_rank", "full_rank", "lstm"], default="low_rank")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--repeats-per-case", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "outputs_short_post"),
    )
    return parser.parse_args(argv)

args = parse_args([
    "--model", "low_rank",
    "--steps", "10000",
    "--hidden-size", "256",
    "--rank", "1",
    "--lr", "0.0001",
])

trainer = Trainer(args)
trainer.train()
