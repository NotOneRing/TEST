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


from copy import deepcopy


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, METHOD_NAME


# RUN_FUNCTION_TEST_SAVE_LOAD = True
RUN_FUNCTION_TEST_SAVE_LOAD = False


from util.torch_to_tf import torch_tensor_item



class TrainGaussianAgent(PreTrainAgent):

    def __init__(self, cfg):
        print("train_gaussian_agent.py: TrainGaussianAgent.__init__()")

        super().__init__(cfg)

        if DEBUG and TEST_LOAD_PRETRAIN:
            self.base_policy_path = cfg.get("base_policy_path", None)

        # self.model.batch_size = self.batch_size


        # Entropy bonus - not used right now since using fixed_std
        self.ent_coef = cfg.train.get("ent_coef", 0)










    def build_ema_model(self, training_flag, item_actions_copy, cond_copy, ent_coef):
        with tf.GradientTape() as tape:
            print("self.model = ", self.model)

            # model_config = self.model.get_config()
            
            # print("model_config = ", model_config)

            # print("train_diffusion_agent.py: run() 4-1")

            # self.ema_model = deepcopy(self.model)
            # self.ema_model = tf.keras.models.clone_model(self.model)

            self.ema_model = tf.keras.models.clone_model(self.model)
            # self.ema_model.network = tf.keras.models.clone_model(self.model.network)


            # print("Self.model trainable weights:")
            # for weight in self.model.trainable_weights:
            #     print(weight.name, weight.shape)

            # print("\nSelf.ema_model trainable weights:")
            # for weight in self.ema_model.trainable_weights:
            #     print(weight.name, weight.shape)


            # print("train_diffusion_agent.py: run() 4-2")

            if DEBUG:
                self.ema_model.loss_ori_t = None
                self.ema_model.p_losses_noise = None
                self.ema_model.call_noise = None
                self.ema_model.call_noise = None
                self.ema_model.call_x = None
                self.ema_model.q_sample_noise = None

            # print("self.ema_model = ", self.ema_model)


            loss_train_ema = self.ema_model.loss_ori(
                training_flag, 
                item_actions_copy, cond_copy, ent_coef)

            # print("item_actions_copy.shape = ", item_actions_copy.shape)
            # print("cond_copy['state'].shape = ", cond_copy['state'].shape)


            # print("loss_train_ema.numpy() = ", loss_train_ema.numpy())


            # print("self.model.trainable_variables = ", self.model.trainable_variables)
            # print("self.ema_model.trainable_variables = ", self.ema_model.trainable_variables)

            # print("self.model.trainable_variables count = ", len(self.model.trainable_variables))
            # print("self.ema_model.trainable_variables count = ", len(self.ema_model.trainable_variables))

            # print("self.model.weights count:", len(self.model.get_weights()))
            # print("self.ema_model.weights count:", len(self.ema_model.get_weights()))


            self.reset_parameters()

            # if DEBUG:
            #     self.take_out_epoch0_weights()





    def debug_gmm_mlp_save_load(self):

        from tensorflow.keras.utils import get_custom_objects

        from model.common.mlp import MLP, ResidualMLP
        
        cur_dict = {
                "MLP": MLP,
                "ResidualMLP": ResidualMLP, 
        }
        get_custom_objects().update(cur_dict)

        print("self.model.network.mlp_weights = ", self.model.network.mlp_weights)
        for var in self.model.network.mlp_weights.moduleList.variables:
            print(f"1:GMM_MLP.mlp_weights: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        import os

        savepath = os.path.join("/ssddata/qtguo/GENERAL_DATA/save_load_test_path/", f"state_{1}.keras")

        tf.keras.models.save_model(self.model.network.mlp_weights, savepath)

        mlp_weights = tf.keras.models.load_model(savepath,  custom_objects=get_custom_objects() )

        # MLP: call() x.shape =  (16, 4)
        
        print("mlp_weights = ", mlp_weights)
        for var in mlp_weights.moduleList.variables:
            print(f"2:GMM_MLP.mlp_weights: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")







    def debug_gmm_save_load(self):

        from tensorflow.keras.utils import get_custom_objects

        from model.common.mlp import MLP, ResidualMLP
        from model.common.mlp_gmm import GMM_MLP
        
        cur_dict = {
                "MLP": MLP,
                "ResidualMLP": ResidualMLP, 
                "GMM_MLP": GMM_MLP
        }
        get_custom_objects().update(cur_dict)

        print("self.model.network = ", self.model.network)
        for var in self.model.network.variables:
            print(f"1:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        num_params = sum(np.prod(var.shape) for var in self.model.network.trainable_variables)
        print(f"1:Number of network parameters: {num_params}")
        print(f"1:Number of network parameters: {sum(var.numpy().size for var in self.model.network.trainable_variables)}")

        import os

        savepath = os.path.join("/ssddata/qtguo/GENERAL_DATA/save_load_test_path/", f"state_{1}.keras")

        tf.keras.models.save_model(self.model.network, savepath)

        network = tf.keras.models.load_model(savepath,  custom_objects=get_custom_objects() )

        num_params = sum(np.prod(var.shape) for var in network.trainable_variables)
        print(f"2:Number of network parameters: {num_params}")
        print(f"2:Number of network parameters: {sum(var.numpy().size for var in network.trainable_variables)}")

        # MLP: call() x.shape =  (16, 4)
        
        print("network = ", network)
        for var in network.variables:
            print(f"2:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")



    def debug_gmm_load_iter(self):

        from tensorflow.keras.utils import get_custom_objects

        from model.common.mlp import MLP, ResidualMLP
        from model.common.mlp_gmm import GMM_MLP
        
        cur_dict = {
                "MLP": MLP,
                "ResidualMLP": ResidualMLP, 
                "GMM_MLP": GMM_MLP
        }
        get_custom_objects().update(cur_dict)

        print("self.model.network = ", self.model.network)
        for var in self.model.network.variables:
            print(f"1:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")

        num_params = sum(np.prod(var.shape) for var in self.model.network.trainable_variables)
        print(f"1:Number of network parameters: {num_params}")
        print(f"1:Number of network parameters: {sum(var.numpy().size for var in self.model.network.trainable_variables)}")

        import os

        base_loadpath = "/ssddata/qtguo/GENERAL_DATA/weights_tensorflow/d3il-pretrain/avoid_m3_pre_gmm_mlp_ta4/2025-02-07_03-56-27_42/checkpoint/state_"

        iter_list = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

        for num in iter_list:
            print("iter_list: epoch = ", num)
            loadpath = base_loadpath + str(num) + "_network.keras"
            network = tf.keras.models.load_model(loadpath,  custom_objects=get_custom_objects() )

            num_params = sum(np.prod(var.shape) for var in network.trainable_variables)
            print(f"2:Number of network parameters: {num_params}")
            print(f"2:Number of network parameters: {sum(var.numpy().size for var in network.trainable_variables)}")

            print("network = ", network)
            for var in network.variables:
                print(f"2:GMM_MLP.network: Layer: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}, var: {var}")


        



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
            
            loss_train_epoch = []
            ent_train_epoch = []

            if DEBUG and _ >= 1:
                break

            for batch_train in self.dataloader_train:



                cond = {}
                cur_actions = batch_train['actions']
                cond['state'] = batch_train["states"]
                cond['rgb'] = batch_train["rgb"]

                item_actions_copy = deepcopy(batch_train['actions'])
                cond_copy = deepcopy(cond)


                if flag:
                    # print("batch_train = ", batch_train)
                    flag = False


                # if self.dataset_train.device == "cpu":
                #     batch_train = batch_to_device(batch_train)

                # self.model.train()
                training_flag = True
                # loss_train = self.model.loss_ori(*batch_train)
                # loss_train.backward()

                # if epoch == 0:
                #     self.build_model(cur_actions, cond)

                with tf.GradientTape() as tape:
                    training_flag=True
                    
                    loss_train, infos_train  = self.model.loss_ori(
                        training_flag, 
                        cur_actions, 
                        cond,
                        ent_coef=self.ent_coef,
                        )



                if epoch == 0:
                    self.build_ema_model(training_flag, item_actions_copy, cond_copy, self.ent_coef)


                if DEBUG and TEST_LOAD_PRETRAIN and epoch == 0:
                    self.load_pickle_network()
                    break
                

                # if DEBUG and epoch == 0:
                #     # self.debug_gmm_save_load()
                #     # self.debug_gmm_mlp_save_load()
                #     # self.debug_gmm_load_iter()
                #     break

                # print("loss_train = ", loss_train)
                # print("loss_train_epoch = ", loss_train_epoch)

                loss_train_epoch.append( torch_tensor_item( loss_train ) )
                ent_train_epoch.append( torch_tensor_item(infos_train["entropy"]) )


                # self.optimizer.step()
                # self.optimizer.zero_grad()

                gradients = tape.gradient(loss_train, self.model.trainable_variables)
                zip_gradients_params = zip(gradients, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip_gradients_params)



                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
                
                epoch += 1

            loss_train = np.mean(loss_train_epoch)
            ent_train = np.mean(ent_train_epoch)


            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                # self.model.eval()
                training_flag = False
                for batch_val in self.dataloader_val:
                    # if self.dataset_val.device == "cpu":
                    #     batch_val = batch_to_device(batch_val)
                    # loss_val, infos_val = self.model.loss(*batch_val)
                    cur_actions = batch_val['actions']
                    cond['state'] = batch_val["states"]
                    cond['rgb'] = batch_val["rgb"]

                    loss_val, infos_val = self.model.loss_ori(training_flag, 
                        cur_actions, cond,
                        ent_coef=self.ent_coef
                        )
                    
                    loss_val_epoch.append( torch_tensor_item(loss_val) )
                # self.model.train()
                training_flag = True
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model(self.epoch)



            # log loss
            if self.epoch % self.log_freq == 0:

                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos_train.items()]
                )
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | {infos_str} | t:{timer():8.4f}"
                )

                # log.info(
                #     f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
                # )
                # if self.use_wandb:
                #     if loss_val is not None:
                #         wandb.log(
                #             {"loss - val": loss_val}, step=self.epoch, commit=False
                #         )
                #     wandb.log(
                #         {
                #             "loss - train": loss_train,
                #             "entropy - train": ent_train,
                #         },
                #         step=self.epoch,
                #         commit=True,
                #     )

            # count
            self.epoch += 1






    # def run(self):

    #     print("train_diffusion_agent.py: TrainDiffusionAgent.run()")

    #     timer = Timer()
    #     self.epoch = 1
    #     cnt_batch = 0

    #     print("self.n_epochs = ", self.n_epochs)


    #     data_before_generator = {
    #         "actions": [],
    #         "states": [],
    #         "rewards": [],
    #         "next_states": [],
    #         "rgb": [],
    #     }



    #     print("self.batch_size = ", self.batch_size)

    #     print("self.n_epochs = ", self.n_epochs)

    #     print("len(self.dataset_train = ", len(self.dataset_train))

    #     print("self.save_model_freq = ", self.save_model_freq)
        
    #     print("len(self.dataset_train) // self.batch_size) = ", len(self.dataset_train) // self.batch_size)

    #     print("(self.save_model_freq * (len(self.dataset_train) // self.batch_size)) = ", (self.save_model_freq * (len(self.dataset_train) // self.batch_size) ) )

    #     print("(self.n_epochs * (len(self.dataset_train) // self.batch_size)) = ", (self.n_epochs * (len(self.dataset_train) // self.batch_size) ) )

    #     for i in range(len(self.dataset_train)):
    #         if DEBUG:
    #             if i == self.batch_size * 10:
    #                 break

    #         batch_train = self.dataset_train[i]
    #         # actions = batch_train.actions
    #         # conditions = batch_train.conditions
    #         # conditions = batch_train['conditions']

    #         actions = batch_train['actions']
    #         data_before_generator['actions'].append(actions)
    #         # states = batch_train['states']
    #         # data_before_generator['states'].append(states)

    #         if "states" in batch_train:
    #             data_before_generator['states'].append(batch_train['states'])
    #         else:
    #             data_before_generator['states'].append(None)

    #         if "rgb" in batch_train:
    #             data_before_generator['rgb'].append(batch_train['rgb'])
    #         else:
    #             data_before_generator['rgb'].append(None)

    #         if "rewards" in batch_train:
    #             data_before_generator['rewards'].append(batch_train['rewards'])
    #         else:
    #             data_before_generator['rewards'].append(None)

    #         if "next_states" in batch_train:
    #             data_before_generator['next_states'].append(batch_train['next_states'])
    #         else:
    #             data_before_generator['next_states'].append(None)

    #     # 构造 Dataset
    #     dataset = tf.data.Dataset.from_tensor_slices(data_before_generator)

    #     buffer_size = len(data_before_generator)
    #     dataset = dataset.shuffle(buffer_size=buffer_size, seed=self.seed)

        
    #     if DEBUG:
    #         self.n_epochs = 2

    #     dataset = dataset.batch(
    #         self.batch_size, drop_remainder=True
    #     ).repeat(self.n_epochs)

    

    #     loss_train_epoch = []

    #     #最终的，但是太慢了，不适合调试网络结构
    #     for epoch, item in enumerate(dataset):



    #         print( f"Epoch {epoch + 1}" )

    #         # continue
            
    #         # # Train
    #         # print(item)
    #         # print("State:", item["states"].numpy())
    #         # print("Action:", item["actions"].numpy())

    #         cond = {}
    #         cond['state'] = item["states"]



    #         with tf.GradientTape() as tape:
    #             # Assuming loss is computed as a callable loss function
    #             # loss_train = self.model.loss(*batch_train, training_flag=True)
    #             # loss_train = self.model.loss(training_flag=True, *batch_train)
    #             training_flag=True
    #             loss_train = self.model.loss(training_flag, item['actions'], cond)

    #         print("self.model.get_config() = ", self.model.get_config())

    #         # # self.ema_model = deepcopy(self.model)

    #         if epoch == 0:
    #             self.ema_model = tf.keras.models.clone_model(self.model)
    #             self.ema_model.set_weights(self.model.get_weights())

    #         gradients = tape.gradient(loss_train, self.model.trainable_variables)
    #         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    #         loss_train_epoch.append(loss_train.numpy())

    #         print("loss_train.numpy() = ", loss_train.numpy())

    #         # Update ema
    #         if cnt_batch % self.update_ema_freq == 0:
    #             self.step_ema()
    #         cnt_batch += 1

    #         loss_train = np.mean(loss_train_epoch)


    #         # # Save model
    #         if epoch % (self.save_model_freq * (len(self.dataset_train) // self.batch_size) ) == 0 or epoch == (self.n_epochs * (len(self.dataset_train) // self.batch_size) - 1 ):
    #             self.save_model()

    #         if DEBUG:
    #             self.save_model()

    #         # Log loss
    #         if epoch % self.log_freq == 0:
    #             log.info(
    #                 f"{epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}"
    #             )








