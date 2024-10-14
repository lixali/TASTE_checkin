import torch
import json
def load_weights(state_dict_path):
        full_state_dict = torch.load(state_dict_path)
        state_dict = {
            k.replace("encoder.encoder", "encoder").replace("module.", ""): v
            for k, v in full_state_dict.items()
        }
        del full_state_dict
        return state_dict


state_dict=load_weights("pretrain_T5/beauty_new_best_sequence_length_3_proposed_pretrain_4_gpus_batch_160_save_step_300_fusin_0_total_data_points/name/best_dev/pytorch_model.bin")


torch.save(state_dict, 'pretrain_T5/beauty_new_best_sequence_length_3_proposed_pretrain_4_gpus_batch_160_save_step_300_fusin_0_total_data_points/name/best_dev/pretrain_weight/pytorch_model.bin')

