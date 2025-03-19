"""
Pre-training diffusion policy

"""


import logging
import numpy as np
import tensorflow as tf

from agent.pretrain.train_agent import PreTrainAgent

log = logging.getLogger(__name__)
from util.timer import Timer


import hydra



import io

from copy import deepcopy


# DEBUG = True
DEBUG = False



class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        print("train_diffusion_agent.py: TrainDiffusionAgent.__init__()")
        super().__init__(cfg)
        self.model.batch_size = self.batch_size


    def build_model(self, cfg):
        # Instantiate the model as a TensorFlow model

        print("train_diffusion_agent.py: TrainDiffusionAgent.build_model()")

        model = hydra.utils.instantiate(cfg.model)

        print("cfg.model.input_shape = ", cfg.model.input_shape)

        model.build(input_shape=(None, *cfg.model.input_shape))  # Ensure the model is built

        #initialize model
        _ = model(tf.constant(np.random.randn(1, *cfg.model.input_shape).astype(np.float32)))

        return model
    

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


        print("self.model = ", self.model)

        print("1self.model.loss = ", self.model.loss)


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

            actions = batch_train['actions']
            data_before_generator['actions'].append(actions)

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

        # construct Dataset
        dataset = tf.data.Dataset.from_tensor_slices(data_before_generator)

        buffer_size = len(data_before_generator)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=self.seed)

        
        if DEBUG:
            self.n_epochs = 2

        dataset = dataset.batch(
            self.batch_size, drop_remainder=True
        ).repeat(self.n_epochs)

    

        loss_train_epoch = []


        for epoch, item in enumerate(dataset):



            print( f"Epoch {epoch + 1}" )


            cond = {}
            cond['state'] = item["states"]

            item_actions_copy = deepcopy(item['actions'])
            cond_copy = deepcopy(cond)


            with tf.GradientTape() as tape:
                training_flag=True
                
                print("item['actions'] = ", item['actions'])
                print("cond = ", cond)
                print("self.model = ", self.model)
                print("self.ema_model = ", self.ema_model)

                print("self.model.loss = ", self.model.loss)

                loss_train = self.model.loss_ori(training_flag, item['actions'], cond)

                print("self.model.network = ", self.model.network)

            gradients = tape.gradient(loss_train, self.model.trainable_variables)


            if epoch == 0:
                with tf.GradientTape() as tape:
                    self.ema_model.network = tf.keras.models.clone_model(self.model.network)
                    print("self.ema_model.network = ", self.ema_model.network)
                    loss_train_ema = self.ema_model.loss_ori(training_flag, item_actions_copy, cond_copy)
                    
                    self.reset_parameters()


            
            if epoch == 0:
                print("self.model.trainable_variables = ", self.model.trainable_variables)
                print("self.ema_model.trainable_variables = ", self.ema_model.trainable_variables)

            zip_gradients_params = zip(gradients, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip_gradients_params)

            loss_train_epoch.append(loss_train.numpy())

            print("loss_train.numpy() = ", loss_train.numpy())

            # Update ema
            if cnt_batch % self.update_ema_freq == 0:
                self.step_ema()
            cnt_batch += 1

            loss_train = np.mean(loss_train_epoch)

            if DEBUG:
                self.save_model()
            # # Save model
            elif epoch % (self.save_model_freq * (len(self.dataset_train) // self.batch_size) ) == 0 or epoch == (self.n_epochs * (len(self.dataset_train) // self.batch_size) - 1 ):
                self.save_model()
            
            if epoch % (len(self.dataset_train) // self.batch_size) == 0:
                self.epoch += 1


            # Log loss
            if epoch % self.log_freq == 0:
                log.info(
                    f"{epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )












