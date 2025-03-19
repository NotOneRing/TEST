"""
Pre-training diffusion policy

"""

import logging
# import wandb
import numpy as np
import tensorflow as tf

from agent.pretrain.train_agent import PreTrainAgent

log = logging.getLogger(__name__)
from util.timer import Timer


import hydra



import io

from copy import deepcopy


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, METHOD_NAME


# RUN_FUNCTION_TEST_SAVE_LOAD = True
RUN_FUNCTION_TEST_SAVE_LOAD = False


from util.torch_to_tf import torch_tensor_item



class TrainDiffusionAgent(PreTrainAgent):

    def __init__(self, cfg):
        print("train_diffusion_agent.py: TrainDiffusionAgent.__init__()")
        super().__init__(cfg)


        print("type(cfg)", type(cfg))

        if DEBUG and TEST_LOAD_PRETRAIN:
            self.base_policy_path = cfg.get("base_policy_path", None)


        self.model.batch_size = self.batch_size


    def take_out_epoch0_weights(self):
        self.epoch0_model_weights = self.model.get_weights()

        self.epoch0_model_network_weights = self.model.network.get_weights()

        self.epoch0_model_network_mlpmean_weights = self.model.network.mlp_mean.get_weights()


        if self.model.network.cond_mlp:
            self.epoch0_model_network_condmlp_weights = self.model.network.cond_mlp.get_weights()


        self.epoch0_model_network_timeemb_weights = self.model.network.time_embedding.get_weights()


        self.epoch0_model_trainable_variables = self.model.trainable_variables

        self.epoch0_model_network_trainable_variables = self.model.network.trainable_variables

        self.epoch0_model_network_mlpmean_trainable_variables = self.model.network.mlp_mean.trainable_variables



        if self.model.network.cond_mlp:
            self.epoch0_model_network_condmlp_trainable_variables = self.model.network.cond_mlp.trainable_variables

        self.epoch0_model_network_timeemb_trainable_variables = self.model.network.time_embedding.trainable_variables



    def test_unchanged_weights(self, epoch):
        epoch_model_weights = self.model.get_weights()
        print("epoch = ", epoch)
        for i in range(len(epoch_model_weights)):
            print("model weights difference = ", np.sum(self.epoch0_model_weights[i] - epoch_model_weights[i]))
            assert np.sum(self.epoch0_model_weights[i] - epoch_model_weights[i]) == 0, "np.sum(epoch0_model_weights[i] - epoch_model_weights[i]) != 0"

        epoch_model_network_weights = self.model.network.get_weights()
        print("epoch = ", epoch)
        for i in range(len(epoch_model_network_weights)):
            print("model network weights difference = ", np.sum(self.epoch0_model_network_weights[i] - epoch_model_network_weights[i]))
            assert np.sum(self.epoch0_model_network_weights[i] - epoch_model_network_weights[i]) == 0, "np.sum(epoch0_model_network_weights[i] - epoch_model_network_weights[i]) != 0"

        epoch_model_network_mlpmean_weights = self.model.network.mlp_mean.get_weights()
        print("epoch = ", epoch)
        for i in range(len(epoch_model_network_mlpmean_weights)):
            print("model network mlpmean weights difference = ", np.sum(self.epoch0_model_network_mlpmean_weights[i] - epoch_model_network_mlpmean_weights[i]))
            assert np.sum(self.epoch0_model_network_mlpmean_weights[i] - epoch_model_network_mlpmean_weights[i]) == 0, "np.sum(epoch0_model_network_mlpmean_weights[i] - epoch_model_network_mlpmean_weights[i]) != 0"



        if self.model.network.cond_mlp:

            epoch_model_network_condmlp_weights = self.model.network.cond_mlp.get_weights()
            print("epoch = ", epoch)
            for i in range(len(epoch_model_network_condmlp_weights)):
                print("model network condmlp weights difference = ", np.sum(self.epoch0_model_network_condmlp_weights[i] - epoch_model_network_condmlp_weights[i]))
                assert np.sum(self.epoch0_model_network_condmlp_weights[i] - epoch_model_network_condmlp_weights[i]) == 0, "np.sum(epoch0_model_network_condmlp_weights[i] - epoch_model_network_condmlp_weights[i]) != 0"

        epoch_model_network_timeemb_weights = self.model.network.time_embedding.get_weights()
        print("epoch = ", epoch)
        for i in range(len(epoch_model_network_timeemb_weights)):
            print("model network timeemb weights difference = ", np.sum(self.epoch0_model_network_timeemb_weights[i] - epoch_model_network_timeemb_weights[i]))
            assert np.sum(self.epoch0_model_network_timeemb_weights[i] - epoch_model_network_timeemb_weights[i]) == 0, "np.sum(epoch0_model_network_timeemb_weights[i] - epoch_model_network_timeemb_weights[i]) != 0"



        print("epoch = ", epoch)
        epoch_model_trainable_variables = self.model.trainable_variables
        for i in range(len(epoch_model_trainable_variables)):
            print("model trainable_variables difference = ", np.sum(self.epoch0_model_trainable_variables[i] - epoch_model_trainable_variables[i]))
            assert np.sum(self.epoch0_model_trainable_variables[i] - epoch_model_trainable_variables[i]) == 0, "np.sum(epoch0_model_trainable_variables[i] - epoch_model_trainable_variables[i]) != 0"

        epoch_model_network_trainable_variables = self.model.network.trainable_variables
        print("epoch = ", epoch)
        for i in range(len(epoch_model_network_trainable_variables)):
            print("model network trainable_variables difference = ", np.sum(self.epoch0_model_network_trainable_variables[i] - epoch_model_network_trainable_variables[i]))
            assert np.sum(self.epoch0_model_network_weights[i] - epoch_model_network_weights[i]) == 0, "np.sum(epoch0_model_network_trainable_variables[i] - epoch_model_network_trainable_variables[i]) != 0"

        epoch_model_network_mlpmean_trainable_variables = self.model.network.mlp_mean.trainable_variables
        print("epoch = ", epoch)
        for i in range(len(epoch_model_network_mlpmean_trainable_variables)):
            print("model network mlpmean trainable_variables difference = ", np.sum(self.epoch0_model_network_mlpmean_trainable_variables[i] - epoch_model_network_mlpmean_trainable_variables[i]))
            assert np.sum(self.epoch0_model_network_mlpmean_trainable_variables[i] - epoch_model_network_mlpmean_trainable_variables[i]) == 0, "np.sum(epoch0_model_network_mlpmean_trainable_variables[i] - epoch_model_network_mlpmean_trainable_variables[i]) != 0"




        if self.model.network.cond_mlp:

            epoch_model_network_condmlp_trainable_variables = self.model.network.cond_mlp.trainable_variables
            print("epoch = ", epoch)
            for i in range(len(epoch_model_network_condmlp_trainable_variables)):
                print("model network condmlp trainable_variables difference = ", np.sum(self.epoch0_model_network_condmlp_trainable_variables[i] - epoch_model_network_condmlp_trainable_variables[i]))
                assert np.sum(self.epoch0_model_network_condmlp_trainable_variables[i] - epoch_model_network_condmlp_trainable_variables[i]) == 0, "np.sum(epoch0_model_network_condmlp_trainable_variables[i] - epoch_model_network_condmlp_trainable_variables[i]) != 0"


        epoch_model_network_timeemb_trainable_variables = self.model.network.time_embedding.trainable_variables
        print("epoch = ", epoch)
        for i in range(len(epoch_model_network_timeemb_trainable_variables)):
            print("model network timeemb trainable_variables difference = ", np.sum(self.epoch0_model_network_timeemb_trainable_variables[i] - epoch_model_network_timeemb_trainable_variables[i]))
            assert np.sum(self.epoch0_model_network_timeemb_trainable_variables[i] - epoch_model_network_timeemb_trainable_variables[i]) == 0, "np.sum(epoch0_model_network_timeemb_trainable_variables[i] - epoch_model_network_timeemb_trainable_variables[i]) != 0"





    def build_model(self, cur_actions, cond):
        with tf.GradientTape() as tape:
            # Assuming loss is computed as a callable loss function
            training_flag=True
            
            loss_train = self.model.loss_ori(
                training_flag, 
                cur_actions, cond)





    def build_ema_model(self, training_flag, item_actions_copy, cond_copy):
        with tf.GradientTape() as tape:
            print("self.model = ", self.model)

            model_config = self.model.get_config()
            
            print("train_diffusion_agent.py: run() 4-1")

            self.ema_model = tf.keras.models.clone_model(self.model)

            print("Self.model trainable weights:")
            for weight in self.model.trainable_weights:
                print(weight.name, weight.shape)

            print("\nSelf.ema_model trainable weights:")
            for weight in self.ema_model.trainable_weights:
                print(weight.name, weight.shape)


            print("train_diffusion_agent.py: run() 4-2")

            if DEBUG:
                self.ema_model.loss_ori_t = None
                self.ema_model.p_losses_noise = None
                self.ema_model.call_noise = None
                self.ema_model.call_noise = None
                self.ema_model.call_x = None
                self.ema_model.q_sample_noise = None

            print("self.ema_model = ", self.ema_model)


            loss_train_ema = self.ema_model.loss_ori(
                training_flag, 
                item_actions_copy, cond_copy)

            print("item_actions_copy.shape = ", item_actions_copy.shape)
            print("cond_copy['state'].shape = ", cond_copy['state'].shape)


            print("loss_train_ema.numpy() = ", loss_train_ema.numpy())


            print("self.model.trainable_variables = ", self.model.trainable_variables)
            print("self.ema_model.trainable_variables = ", self.ema_model.trainable_variables)

            print("self.model.trainable_variables count = ", len(self.model.trainable_variables))
            print("self.ema_model.trainable_variables count = ", len(self.ema_model.trainable_variables))

            print("self.model.weights count:", len(self.model.get_weights()))
            print("self.ema_model.weights count:", len(self.ema_model.get_weights()))


            self.reset_parameters()

            if DEBUG:
                self.take_out_epoch0_weights()






    def debug_diffusion_img_save_load(self):


        if hasattr(self.model.network, 'compress'):
            self.save_load_params("self.model.network.compress", self.model.network.compress)

        if hasattr(self.model.network, 'compress1'):
            self.save_load_params("self.model.network.compress1", self.model.network.compress1)

        if hasattr(self.model.network, 'compress2'):
            self.save_load_params("self.model.network.compress2", self.model.network.compress2)

        if hasattr(self.model.network, 'mlp_logvar'):
            self.save_load_params("self.model.network.logvar", self.model.network.logvar)

        if hasattr(self.model.network, 'mlp_mean'):
            self.save_load_params("self.model.network.mlp_mean", self.model.network.mlp_mean)

        if hasattr(self.model.network, 'backbone'):
            self.save_load_params("self.model.network.backbone", self.model.network.backbone)


        if hasattr(self.model.network, 'time_embedding'):
            self.save_load_params("self.model.network.time_embedding", self.model.network.time_embedding)







    def run(self):

        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        epoch = 0
        for _ in range(self.n_epochs):
            print("epoch = ", _)
            flag = True
            # train

            training_flag = True

        
            if DEBUG and TEST_LOAD_PRETRAIN and _ == 1:
                break

            loss_train_epoch = []
            for batch_train in self.dataloader_train:

                cond = {}
                cur_actions = batch_train['actions']
                cond['state'] = batch_train["states"]
                if 'rgb' in batch_train:
                    cond['rgb'] = batch_train["rgb"]

                item_actions_copy = deepcopy(batch_train['actions'])
                cond_copy = deepcopy(cond)


                if flag:
                    flag = False

                training_flag = True


                with tf.GradientTape() as tape:
                    training_flag=True
                    
                    loss_train = self.model.loss_ori(
                        training_flag, 
                        cur_actions, cond)


                if epoch == 0:
                    self.build_ema_model(training_flag, item_actions_copy, cond_copy)


                if DEBUG and TEST_LOAD_PRETRAIN and epoch == 0:
                    self.load_pickle_network()
                    self.debug_diffusion_img_save_load()
                    break




                loss_train_epoch.append( torch_tensor_item( loss_train ))

                gradients = tape.gradient(loss_train, self.model.trainable_variables)
                zip_gradients_params = zip(gradients, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip_gradients_params)



                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
                
                epoch += 1

            loss_train = np.mean(loss_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                training_flag = False
                for batch_val in self.dataloader_val:
                    cur_actions = batch_val['actions']
                    cond['state'] = batch_val["states"]

                    if 'rgb' in batch_val:
                        cond['rgb'] = batch_val["rgb"]

                    loss_val = self.model.loss_ori(training_flag, 
                        cur_actions, cond)
                    
                    loss_val_epoch.append( torch_tensor_item(loss_val) )
                training_flag = True
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model(self.epoch)

            # log loss
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                )


            self.epoch += 1



