"""
Pre-training diffusion policy

"""


import logging
import wandb
import numpy as np
import tensorflow as tf

from agent.pretrain.train_agent import PreTrainAgent, batch_to_device

log = logging.getLogger(__name__)
from util.timer import Timer


import hydra


class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        print("train_diffusion_agent.py: TrainDiffusionAgent.__init__()", flush=True)
        super().__init__(cfg)

        # # # Use tf's model handling
        # self.model = self.build_model(cfg)

    def build_model(self, cfg):
        # Instantiate the model as a TensorFlow model

        print("train_diffusion_agent.py: TrainDiffusionAgent.build_model()", flush=True)

        model = hydra.utils.instantiate(cfg.model)
        model.build(input_shape=(None, *cfg.model.input_shape))  # Ensure the model is built
        return model
    

    def run(self):

        print("train_diffusion_agent.py: TrainDiffusionAgent.run()", flush=True)

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0

        print("self.n_epochs = ", self.n_epochs)

        for epoch_index in range(self.n_epochs):

            print("epoch = ", epoch_index)

            # Train
            loss_train_epoch = []
            for batch_train in self.dataloader_train:

                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                print("self.model = ", self.model)

                # Enable training mode
                # self.model.train()

                with tf.GradientTape() as tape:
                    # Assuming loss is computed as a callable loss function
                    loss_train = self.model.loss(*batch_train, training_flag=True)

                gradients = tape.gradient(loss_train, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                loss_train_epoch.append(loss_train.numpy())

                # Update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)

            # Validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(*batch_val, training_flag=False)
                    loss_val_epoch.append(loss_val.numpy())
                self.model.train()

            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # Update lr scheduler
            self.lr_scheduler.step()

            # Save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # Log loss
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

            # Increment epoch count
            self.epoch += 1
































