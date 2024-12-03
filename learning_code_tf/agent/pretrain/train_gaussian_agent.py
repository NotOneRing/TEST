"""
Pre-training Gaussian/GMM policy

"""

import logging
import wandb
import numpy as np

import tensorflow as tf

log = logging.getLogger(__name__)
from util.timer import Timer

from agent.pretrain.train_agent import PreTrainAgent

# , batch_to_device

# DEBUG = True
DEBUG = False


class TrainGaussianAgent(PreTrainAgent):

    def __init__(self, cfg):
        print("train_gaussian_agent.py: TrainGaussianAgent.__init__()")

        super().__init__(cfg)

        # Entropy bonus - not used right now since using fixed_std
        self.ent_coef = cfg.train.get("ent_coef", 0)



    def run(self):

        print("train_diffusion_agent.py: TrainDiffusionAgent.run()")

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0

        print("self.n_epochs = ", self.n_epochs)


        data_before_generator = {
            "actions": [],
            "states": [],
            "rewards": [],
            "next_states": [],
            "rgb": [],
        }



        print("self.batch_size = ", self.batch_size)

        print("self.n_epochs = ", self.n_epochs)

        print("len(self.dataset_train = ", len(self.dataset_train))

        print("self.save_model_freq = ", self.save_model_freq)
        
        print("len(self.dataset_train) // self.batch_size) = ", len(self.dataset_train) // self.batch_size)

        print("(self.save_model_freq * (len(self.dataset_train) // self.batch_size)) = ", (self.save_model_freq * (len(self.dataset_train) // self.batch_size) ) )

        print("(self.n_epochs * (len(self.dataset_train) // self.batch_size)) = ", (self.n_epochs * (len(self.dataset_train) // self.batch_size) ) )

        for i in range(len(self.dataset_train)):
            if DEBUG:
                if i == self.batch_size * 10:
                    break

            batch_train = self.dataset_train[i]
            # actions = batch_train.actions
            # conditions = batch_train.conditions
            # conditions = batch_train['conditions']

            actions = batch_train['actions']
            data_before_generator['actions'].append(actions)
            # states = batch_train['states']
            # data_before_generator['states'].append(states)

            if "states" in batch_train:
                data_before_generator['states'].append(batch_train['states'])
            else:
                data_before_generator['states'].append(None)

            if "rgb" in batch_train:
                data_before_generator['rgb'].append(batch_train['rgb'])
            else:
                data_before_generator['rgb'].append(None)

            if "rewards" in batch_train:
                data_before_generator['rewards'].append(batch_train['rewards'])
            else:
                data_before_generator['rewards'].append(None)

            if "next_states" in batch_train:
                data_before_generator['next_states'].append(batch_train['next_states'])
            else:
                data_before_generator['next_states'].append(None)

        # 构造 Dataset
        dataset = tf.data.Dataset.from_tensor_slices(data_before_generator)

        buffer_size = len(data_before_generator)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=self.seed)

        
        if DEBUG:
            self.n_epochs = 2

        dataset = dataset.batch(
            self.batch_size, drop_remainder=True
        ).repeat(self.n_epochs)

    

        loss_train_epoch = []

        #最终的，但是太慢了，不适合调试网络结构
        for epoch, item in enumerate(dataset):



            print( f"Epoch {epoch + 1}" )

            # continue
            
            # # Train
            # print(item)
            # print("State:", item["states"].numpy())
            # print("Action:", item["actions"].numpy())

            cond = {}
            cond['state'] = item["states"]



            with tf.GradientTape() as tape:
                # Assuming loss is computed as a callable loss function
                # loss_train = self.model.loss(*batch_train, training_flag=True)
                # loss_train = self.model.loss(training_flag=True, *batch_train)
                training_flag=True
                loss_train = self.model.loss(training_flag, item['actions'], cond)

            print("self.model.get_config() = ", self.model.get_config())

            # # self.ema_model = deepcopy(self.model)

            if epoch == 0:
                self.ema_model = tf.keras.models.clone_model(self.model)
                self.ema_model.set_weights(self.model.get_weights())

            gradients = tape.gradient(loss_train, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            loss_train_epoch.append(loss_train.numpy())

            print("loss_train.numpy() = ", loss_train.numpy())

            # Update ema
            if cnt_batch % self.update_ema_freq == 0:
                self.step_ema()
            cnt_batch += 1

            loss_train = np.mean(loss_train_epoch)


            # # Save model
            if epoch % (self.save_model_freq * (len(self.dataset_train) // self.batch_size) ) == 0 or epoch == (self.n_epochs * (len(self.dataset_train) // self.batch_size) - 1 ):
                self.save_model()

            if DEBUG:
                self.save_model()

            # Log loss
            if epoch % self.log_freq == 0:
                log.info(
                    f"{epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )








