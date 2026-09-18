import os

import torch
import torch.nn as nn
import torch.nn.functional as functional
from pytorch_lightning import LightningModule, Callback
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sn


class GradientLoggerCallback(Callback):
    def __init__(self):
        super(GradientLoggerCallback, self).__init__()
        self.gradient_history = dict()

    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.gradient_history.keys():
                    self.gradient_history[name] = []
                self.gradient_history[name].append(param.grad.detach().cpu().numpy())
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        self.gradient_history = dict()


class ConfusionMatrixPlotterCallback(Callback):
    def __init__(self, log_path, absolute_values=True):
        super(ConfusionMatrixPlotterCallback, self).__init__()
        self.log_path = log_path
        self.absolute_values = absolute_values

        self.last_best_val_acc = 0
        self.last_best_train_acc = 0
        self.last_best_test_acc = 0

        self.quantized_model = False

    def on_validation_end(self, trainer, pl_module):
        mode = ["val", "train"]
        self.plot_cm(mode, trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        mode = ["test"]
        self.plot_cm(mode, trainer, pl_module)

    def plot_cm(self, mode, trainer, pl_module):
        for m in mode:
            try:
                acc = trainer.logged_metrics[f"{m}_acc"]
                last_acc = getattr(self, f"last_best_{m}_acc")
                cm = getattr(pl_module, f"{m}_cm")
                if acc > last_acc:
                    setattr(self, f"last_best_{m}_acc", acc)
                    plt.clf()
                    if not self.absolute_values:
                        cm = cm / cm.sum(
                            dim=0, keepdim=True
                        )  # sum along true class axis
                    detached_cm = cm.detach().cpu().numpy()
                    if self.absolute_values:
                        sn.heatmap(
                            detached_cm,
                            annot=True,
                            fmt=".0f",
                            annot_kws={"size": 7, "weight": "normal", "color": "blue"},
                        )
                    else:
                        sn.heatmap(
                            detached_cm,
                            annot=True,
                            fmt=".2f",
                            annot_kws={"size": 7, "weight": "normal", "color": "blue"},
                        )
                    if not pl_module.include_null_class:
                        plt.xlabel("predicted class")
                        plt.ylabel("true class")
                    else:
                        plt.xlabel("predicted class (including null class)")
                        plt.ylabel("true class (including null class)")
                        if isinstance(
                            trainer.logged_metrics[
                                f"{m}_acc_binary_classification_null_class"
                            ],
                            torch.Tensor,
                        ):
                            acc_binary_classification_null_class = (
                                trainer.logged_metrics[
                                    f"{m}_acc_binary_classification_null_class"
                                ].item()
                            )
                            acc_binary_classification_other_classes = (
                                trainer.logged_metrics[
                                    f"{m}_acc_binary_classification_other_classes"
                                ].item()
                            )
                        else:
                            acc_binary_classification_null_class = (
                                trainer.logged_metrics[
                                    f"{m}_acc_binary_classification_null_class"
                                ]
                            )
                            acc_binary_classification_other_classes = (
                                trainer.logged_metrics[
                                    f"{m}_acc_binary_classification_other_classes"
                                ]
                            )
                        plt.text(
                            0.8,
                            1.1,
                            f"null_class_acc: {str(acc_binary_classification_null_class)[:6]}\n other_classes_acc: {str(acc_binary_classification_other_classes)[:6]}",
                            transform=plt.gca().transAxes,
                            fontsize=8,
                            verticalalignment="top",
                            horizontalalignment="left",
                        )
                    if isinstance(acc, float):
                        plt.title(
                            f'{m}_acc={str(trainer.logged_metrics[f"{m}_acc"])[:6]}'
                        )
                    else:
                        plt.title(
                            f'{m}_acc={str(trainer.logged_metrics[f"{m}_acc"].item())[:6]}'
                        )
                    plt.tight_layout()
                    suffix = "_compressed" if self.quantized_model else ""
                    plt.savefig(
                        self.log_path + f"/{m}_confusion_matrix{suffix}.png",
                        bbox_inches="tight",
                    )
                # reset of confusion matrix
                cm.zero_()
            except KeyError:  # this case is important for the Sanity Check
                pass


class BestValAccCallback(Callback):
    def __init__(self):
        super(BestValAccCallback, self).__init__()
        self.last_best_val_acc = 0

    def on_validation_end(self, trainer, pl_module):
        val_acc = trainer.logged_metrics["val_acc"]
        if val_acc > self.last_best_val_acc:
            self.last_best_val_acc = val_acc


def plot_output(a_targets, b_preds, logdir, name="train_results_fig", interpol_steps=8):
    # plot targets and preds on same figure
    plt.plot(
        a_targets[:, 0],
        a_targets[:, 1],
        label="target",
        ls="--",
        marker=".",
    )
    plt.plot(
        b_preds[:, 0],
        b_preds[:, 1],
        label="pred",
        alpha=0.75,
    )
    for i in range(int(b_preds.shape[0] / interpol_steps)):
        plt.plot(
            b_preds[interpol_steps * i, 0],
            b_preds[interpol_steps * i, 1],
            ls="",
            marker=".",
            c="red",
        )
    plt.legend()
    plt.savefig(os.path.join(logdir, name))
    plt.close()


class SpikingNetwork(LightningModule):
    def __init__(
        self,
        net,
        lr=1e-3,
        spike_regu=0.0,
        target_spike_share=0.0,
        weight_regu=0.0,
        hyperparameters=None,
        output_decoder="avg",
        lr_scheduled=False,
        lr_decay=0.1,
        lr_update_freq=10,
        l2_regu=0.0,
    ):
        super().__init__()
        self.net = net
        self.lr = lr
        self.spike_regu = spike_regu
        self.target_spike_share = target_spike_share
        self.weight_regu = weight_regu
        self.output_decoder = output_decoder
        self.lr_scheduled = lr_scheduled
        self.lr_decay = lr_decay
        self.lr_update_freq = lr_update_freq
        self.l2_regu = l2_regu

        if hyperparameters is not None:
            self.save_hyperparameters(hyperparameters)

        if self.net.hyperparameters["dataset_name"] == "NeuroMorse":
            self.include_null_class = True
            self.null_class_decoder = self.null_class_decoder_fn
            self.register_buffer("null_class_thr", torch.zeros((self.net.out_shape,)))
        else:
            self.include_null_class = False
            self.null_class_decoder = torch.nn.Identity()
        self.out_shape = self.net.out_shape + self.include_null_class

        self.cce = torch.nn.functional.cross_entropy
        self.acc = MulticlassAccuracy(num_classes=self.out_shape, top_k=1)
        self.best_val_acc = 0.0

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.out_shape)
        self.train_cm = torch.zeros(size=(self.out_shape, self.out_shape))
        self.val_cm = torch.zeros(size=(self.out_shape, self.out_shape))
        self.test_cm = torch.zeros(size=(self.out_shape, self.out_shape))

        if self.output_decoder == "last":
            self.output_decoder_fn = self.output_decoder_last
        elif self.output_decoder == "avg":
            self.output_decoder_fn = self.output_decoder_mean
        elif self.output_decoder == "max":
            self.output_decoder_fn = self.output_decoder_max
        elif self.output_decoder == "train":
            self.decoder_layer = nn.Parameter(
                torch.normal(
                    mean=0.0,
                    std=0.05,
                    size=(hyperparameters["seq_len"], 1, hyperparameters["out_shape"]),
                )
            )
            self.output_decoder_fn = self.output_decoder_trainable
        elif "sum:" in self.output_decoder:
            _, output_drop_str = self.output_decoder.split(":")
            self.output_drop = int(output_drop_str)
            self.output_decoder_fn = self.output_decoder_partial_sum
        else:
            raise Exception(f"Output decoder type {self.output_decoder} is no known!")

    def null_class_decoder_fn(self, pred):
        certainty = (pred >= self.null_class_thr.unsqueeze(0)).sum(dim=1) != 0
        null_class_output = (
            self.null_class_thr.max() * ~certainty - 1e6 * certainty
        )  # 1e6 as unreasonably low value that would never be undercut by any max prediction value

        include_null_class_output = torch.cat(
            (pred, null_class_output.unsqueeze(1)), dim=1
        )
        return include_null_class_output

    @staticmethod
    def calc_acc(pred: torch.Tensor, label: torch.Tensor):
        """preds and labels are given as indices and are NOT one-hot-encoded"""
        return torch.sum(pred == label) / label.numel()

    def sparsity_loss(self, preds, targets):
        """preds is the spk dictionary containing all layer spikes outputs."""
        spikes = preds

        total_spikes = 0
        spike_share = 0
        num_el = 1
        if isinstance(spikes, list):
            num_el = len(spikes)
        else:
            spikes = [spikes]
        batch_size = spikes[0].shape[-2]  # -2 is idx of batch_size
        for i in range(num_el):
            total_spikes = total_spikes + torch.sum(spikes[i])
            spike_share = spike_share + total_spikes / (
                torch.numel(spikes[i]) * num_el + 1
            )
        spike_loss = torch.nn.functional.l1_loss(
            spike_share, torch.tensor(self.target_spike_share).to(spike_share.device)
        )

        avg_spikes = total_spikes / batch_size

        return spike_loss, avg_spikes

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, p=2) ** 2
        return l2_loss

    def forward(self, x):
        result, spikes = self.net(x)
        decoded_result = self.output_decoder_fn(result)

        return decoded_result, spikes

    def base_step(self, batch):
        x, y_int = batch
        self.net.testing = False

        y_hat, spikes = self.net(x)
        y_hat = self.output_decoder_fn(y_hat)
        y_hat = self.null_class_decoder(y_hat)

        y = functional.one_hot(y_int, num_classes=(self.out_shape)).float()

        cce_loss = self.cce(y_hat, y)
        spike_loss, avg_spikes = self.sparsity_loss(spikes, y)
        l2_loss = self.l2_regularization_loss()
        loss = cce_loss + self.spike_regu * spike_loss + self.l2_regu * l2_loss

        accuracies = dict()
        accuracies["acc"] = self.calc_acc(torch.argmax(y_hat, dim=-1), y_int)
        if self.include_null_class:
            null_class_mask = y_int == self.out_shape - 1

            y_hat_binary_classification = (
                torch.argmax(y_hat, dim=-1) == self.out_shape - 1
            ).to(torch.int)
            y_int_binary_classification = null_class_mask.to(torch.int)

            accuracies["binary_classification_null_class"] = self.calc_acc(
                y_hat_binary_classification[null_class_mask],
                y_int_binary_classification[null_class_mask],
            )
            accuracies["binary_classification_other_classes"] = self.calc_acc(
                y_hat_binary_classification[~null_class_mask],
                y_int_binary_classification[~null_class_mask],
            )

        return loss, cce_loss, spike_loss, accuracies, avg_spikes, y_hat, y_int

    def training_step(self, batch, batch_idx):
        loss, cce_loss, spike_loss, accuracies, avg_spikes, y_hat, y_int = (
            self.base_step(batch)
        )
        acc = accuracies["acc"]

        self.train_cm = self.train_cm + self.confusion_matrix(
            torch.argmax(y_hat, dim=-1), y_int
        ).to(self.train_cm.device)

        if self.include_null_class:
            pred_idx = torch.argmax(y_hat, dim=1)
            for i in range(self.out_shape - 1):
                max_preds = y_hat[pred_idx == i, i]
                if max_preds.numel() > 0:
                    self.null_class_thr[i] = (
                        max_preds.min().detach()
                    )  # detaching so it is not included in the gradient computation

        results = {
            "loss": loss,
            "cce_loss": cce_loss,
            "spike_loss": spike_loss,
            "acc": acc,
            "avg_spikes": avg_spikes,
            "y_hat": y_hat,
            "y_int": y_int,
        }
        if self.include_null_class:
            results["binary_classification_null_class"] = accuracies[
                "binary_classification_null_class"
            ]
            results["binary_classification_other_classes"] = accuracies[
                "binary_classification_other_classes"
            ]
        self.log_results("train", results)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, cce_loss, spike_loss, accuracies, avg_spikes, y_hat, y_int = (
            self.base_step(batch)
        )
        acc = accuracies["acc"]

        self.val_cm = self.val_cm + self.confusion_matrix(
            torch.argmax(y_hat, dim=-1), y_int
        ).to(self.val_cm.device)

        results = {
            "loss": loss,
            "cce_loss": cce_loss,
            "spike_loss": spike_loss,
            "acc": acc,
            "avg_spikes": avg_spikes,
            "y_hat": y_hat,
            "y_int": y_int,
        }
        if self.include_null_class:
            results["binary_classification_null_class"] = accuracies[
                "binary_classification_null_class"
            ]
            results["binary_classification_other_classes"] = accuracies[
                "binary_classification_other_classes"
            ]
        self.log_results("val", results)

        return loss

    def test_step(self, batch, batch_idx):
        loss, cce_loss, spike_loss, accuracies, avg_spikes, y_hat, y_int = (
            self.base_step(batch)
        )
        acc = accuracies["acc"]

        self.test_cm = self.test_cm + self.confusion_matrix(
            torch.argmax(y_hat, dim=-1), y_int
        ).to(self.test_cm.device)

        results = {
            "loss": loss,
            "cce_loss": cce_loss,
            "spike_loss": spike_loss,
            "acc": acc,
            "avg_spikes": avg_spikes,
            "y_hat": y_hat,
            "y_int": y_int,
        }
        if self.include_null_class:
            results["binary_classification_null_class"] = accuracies[
                "binary_classification_null_class"
            ]
            results["binary_classification_other_classes"] = accuracies[
                "binary_classification_other_classes"
            ]
        self.log_results("test", results)

        return loss

    def log_results(self, mode, results):
        self.log(
            f"{mode}_loss",
            results["loss"].detach(),
            batch_size=results["y_hat"].shape[0],
        )
        self.log(
            f"{mode}_cce_loss",
            results["cce_loss"].detach(),
            batch_size=results["y_hat"].shape[0],
        )
        self.log(
            f"{mode}_spike_loss",
            results["spike_loss"].detach(),
            batch_size=results["y_hat"].shape[0],
        )
        self.log(
            f"{mode}_acc", results["acc"].detach(), batch_size=results["y_hat"].shape[0]
        )
        self.log(
            f"{mode}_avg_spikes",
            results["avg_spikes"].detach(),
            batch_size=results["y_hat"].shape[0],
        )
        if self.include_null_class:
            self.log(
                f"{mode}_acc_binary_classification_null_class",
                results["binary_classification_null_class"].detach(),
                batch_size=results["y_hat"].shape[0],
            )
            self.log(
                f"{mode}_acc_binary_classification_other_classes",
                results["binary_classification_other_classes"].detach(),
                batch_size=results["y_hat"].shape[0],
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_regu
        )

        if not self.lr_scheduled:
            return optimizer
        elif self.lr_scheduled == "lin":
            scheduler = StepLR(
                optimizer, step_size=self.lr_update_freq, gamma=self.lr_decay
            )
        elif self.lr_scheduled == "cos":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.lr_update_freq, eta_min=self.lr / self.lr_decay
            )  # T_max is the number of epochs after which the minimum lr is reached
        elif (
            self.lr_scheduled
        ):  # have to include True and False as possible values for backwards-compatability reasons
            scheduler = StepLR(
                optimizer, step_size=self.lr_update_freq, gamma=self.lr_decay
            )
        else:
            raise Exception(f"LR scheduler type {self.lr_scheduled} not known..")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def output_decoder_last(self, pred):
        return pred[-1, :, :]

    def output_decoder_mean(self, pred):
        return torch.mean(pred, dim=0)

    def output_decoder_max(self, pred):
        return torch.max(pred, dim=0)[0]

    def output_decoder_trainable(self, pred):
        weighted_output = pred * self.decoder_layer
        return torch.mean(weighted_output, dim=0)

    def output_decoder_partial_sum(self, pred):
        return torch.sum(pred[self.output_drop :], dim=0)
