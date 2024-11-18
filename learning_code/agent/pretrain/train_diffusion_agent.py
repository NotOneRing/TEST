"""
Pre-training diffusion policy

"""

import logging
import wandb
import numpy as np

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        print("train_diffusion_agent.py: TrainDiffusionAgent.__init__()", flush = True)
        super().__init__(cfg)

    def run(self):

        print("train_diffusion_agent.py: TrainDiffusionAgent.run()", flush = True)

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0

        print("self.n_epochs = ", self.n_epochs)

        # for _ in range(self.n_epochs):
        for epoch_index in range(self.n_epochs):

            print("epoch = ", epoch_index)

            # print("type(self.dataloader_train) = ", type(self.dataloader_train))
            # print("(self.dataloader_train).shape = ", (self.dataloader_train).shape)

            # train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:

                # print("batch_train = ", batch_train)

                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                print("self.model = ", self.model)
                
                # model.eval()  # train() is to change the model's mode
                # print(model.training)  # False
                # model.train()  # train is to change the model's mode
                # print(model.training)  # True
                # for example, in some cases, train is to open Dropout, eval is to close Dropout

                self.model.train()

                loss_train = self.model.loss(*batch_train)

                # Following condition to start loss.backward()：
                # 1.forward() with PyTorch Tensor
                # 2. Support auto differentiation in used operators and provide gradient propagation rule.
                # 3. loss is a scalar in the forward() calculation。
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())

                # step() 是优化器的核心方法，用于 根据梯度更新模型的参数。
                # 梯度在调用 loss.backward() 时已经被计算并存储在每个参数的 .grad 属性中。
                # step() 的具体行为取决于优化器的类型（如 SGD、Adam 等）。
                # 工作流程：
                # 遍历模型中的所有参数。
                # 使用每个参数的当前值和梯度，按照优化算法（如 SGD 的梯度下降公式）计算参数的更新值。
                # 更新参数值。
                self.optimizer.step()


                # 清除所有参数的梯度缓存（grad 属性置为零）。
                # 避免梯度在每次 loss.backward() 时累积（PyTorch 默认会累积梯度）。
                # 为什么需要清除梯度？
                # 梯度累积行为： PyTorch 默认会将每次调用 loss.backward() 计算出的梯度累加到已有的梯度中。这对于某些场景（如梯度累积训练）是有意为之，但在一般的训练过程中，这可能导致梯度计算错误。

                # 如果不调用 zero_grad()：

                # 假设前一次梯度是 grad_prev，后一次梯度是 grad_new。
                # 每次更新的梯度会是 grad = grad_prev + grad_new，从而影响优化效果。
                self.optimizer.zero_grad()

                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(*batch_val)
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1































