import os
import torch
import h5py
import pdb

import numpy as np
import open3d as o3d

from data_utils.utils import pad_sequence, generate_point_cloud_from_depth
from torch.utils.data.dataloader import default_collate
from scipy.spatial.transform import Rotation as R

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        target_key,
        seq_length=1,
        pad_seq_length=True,
        filter_by_attribute=None,
        image_size=None,
        hdf5_normalize_obs=False,
        obs_info_keys=None,
    ):
        super(SequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self._hdf5_file = None

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)
        self.obs_info_keys = None
        if obs_info_keys is not None:
            self.obs_info_keys = obs_info_keys

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.image_size = image_size

        self.pad_seq_length = pad_seq_length

        self.filter_by_attribute = filter_by_attribute

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        self.counter = 0

        self.num_zeros = 0
        self.num_ones = 0
        self.get_target_distribution(target_key=target_key)

        # self.obs_normalization_stats = None
        # if self.hdf5_normalize_obs:
        #     self.obs_normalization_stats = self.normalize_obs()

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file
    
    def get_target_distribution(self, target_key):
        """
        Calculate the distribution of contact labels (0s and 1s) across all demos.
        
        Returns:
            tuple: (num_zeros, num_ones) - Count of 0s and 1s in the dataset
        """
        
        for demo_id in self.demos:
            # Skip episodes with only one entry for contacts
            if len(self.hdf5_file[f"data/{demo_id}/extras/{target_key}"]) == 1:
                continue
                
            target = np.array(self.hdf5_file[f"data/{demo_id}/extras/{target_key}"])
            # Count from index 1 onwards since we use the next timestep as label
            self.num_zeros += np.sum(target[1:] == 0)
            self.num_ones += np.sum(target[1:] == 1)
                    
    def load_demo_info(self, filter_by_attribute=None, demos=None):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

            demos (list): list of demonstration keys to load from the hdf5 file. If 
                omitted, all demos in the file (or under the @filter_by_attribute 
                filter key) are used.
        """
        # filter demo trajectory by mask
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            self.demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
        else:
            self.demos = list(self.hdf5_file['data'].keys())

        # sort demo keys
        inds = np.argsort([int(elem[8:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:

            # Skip episodes with only one entry for contacts (only the starting timestep) i.e. this datapoint has no label
            if len(self.hdf5_file["data/{}".format(ep)]['extras']['contacts']) == 1:
                continue

            # Skip episodes where robot went crazy (for now I am chcking this by checking if camera pos is weird)
            camera_pos = np.array(self.hdf5_file[f'data/{ep}/proprioceptions/camera_qpos'])[0][1]
            if camera_pos > 0.3:
                continue

            # demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            demo_length = len(self.hdf5_file["data/{}".format(ep)]['actions']['actions'])
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            # # Added by arpit
            # num_sequences = 1

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                if num_sequences < 1:
                    print(f"Sequence {ep} can't be used with this sequence length")
                    #assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            # print("num_sequences: ", num_sequences)
            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    

    def extract_observations_info_from_hdf5(self, obs_info_strings, obs_info_shapes):
        # Reconstruct original structure
        idx = 0
        reconstructed_data = []
        for shape in obs_info_shapes:
            sublist = []
            for _ in range(shape):
                sublist.append(list(map(lambda x: x.decode('utf-8'), obs_info_strings[idx:idx+2])))
                idx += 2
            reconstructed_data.append(sublist)

        reconstructed_data = np.array(reconstructed_data, dtype=object)
        # for i in range(len(reconstructed_data)):
        #     print(i, np.array(reconstructed_data[i]).shape)
        #     if i == 0:
        #         print(reconstructed_data)
        return reconstructed_data
    
    def get_seg_semantic_info(self, ep):
        # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id
        if 'seg_semantic_strings' in self.hdf5_file[f'data/{ep}/observations_info'].keys():
            seg_semantic_strings = np.array(self.hdf5_file["data/{}/observations_info/seg_semantic_strings".format(ep)])
            seg_semantic_shapes = np.array(self.hdf5_file["data/{}/observations_info/seg_semantic_shapes".format(ep)])
            seg_semantic = self.extract_observations_info_from_hdf5(obs_info_strings=seg_semantic_strings, 
                                                                        obs_info_shapes=seg_semantic_shapes)
            # print("111: ", seg_semantic.shape)
        else:
            hd5key = "data/{}/observations_info/seg_semantic".format(ep)
            # seg_semantic = self.hdf5_file[hd5key]
            seg_semantic = np.array(self.hdf5_file[hd5key]).astype(str)
            # print("222: ", seg_semantic.shape)
        return seg_semantic
    
    def get_seg_instance_info(self, ep):
        # Basically dealing with HDF5 limitation: handling inconsistent length arrays in observations_info/seg_instance_id
        if 'seg_instance_strings' in self.hdf5_file[f'data/{ep}/observations_info'].keys():
            seg_instance_strings = np.array(self.hdf5_file["data/{}/observations_info/seg_instance_strings".format(ep)])
            seg_instance_shapes = np.array(self.hdf5_file["data/{}/observations_info/seg_instance_shapes".format(ep)])
            seg_instance = self.extract_observations_info_from_hdf5(obs_info_strings=seg_instance_strings, 
                                                                        obs_info_shapes=seg_instance_shapes)
            # print("111: ", seg_instance.shape)
        else:
            hd5key = "data/{}/observations_info/seg_instance".format(ep)
            # seg_instance = hdf5_file[hd5key]
            seg_instance = np.array(self.hdf5_file[hd5key]).astype(str)
            # print("222: ", seg_instance.shape)
        return seg_instance
    
    def get_pcd(self, ep):
        depth = self.hdf5_file[f"data/{ep}/observations/depth"]
        intr =  np.array([
            [103.8416,   0.0000,  64.0000],
            [  0.0000, 103.8416,  64.0000],
            [  0.0000,   0.0000,   1.0000]])
        
        
        extrinsic_matrix = np.array(self.hdf5_file["data"][ep]["proprioceptions"]["extrinsic_matrix"])[0]

        # creating mask to remove floors
        seg_semantic = self.hdf5_file[f'data/{ep}/observations/seg_semantic']
        seg_instance = self.hdf5_file[f'data/{ep}/observations/seg_instance']

        # seg_semantic_info = self.get_seg_semantic_info(ep)
        seg_instance_info = self.get_seg_instance_info(ep)
        # seq_num = 0

        pcd_points = []
        pcd_normals = []
        pcd_colors = []
        for seq_num in range(len(depth)):

            # creating mask to remove floors
            floor_id = -1
            # for row in seg_semantic_info[seq_num]:
            for row in seg_instance_info[seq_num]:
                sem_id, class_name = int(row[0]), row[1]
                # if class_name == 'floors':
                if class_name == 'groundPlane':
                    floor_id = sem_id
                    break

            if floor_id != -1:
                mask = np.zeros_like(depth[seq_num])
                # mask[seg_semantic[seq_num] != floor_id] = 1
                mask[seg_instance[seq_num] != floor_id] = 1
            else:
                mask = np.ones_like(depth[seq_num])

            o3d_pcd = generate_point_cloud_from_depth(depth[seq_num], intr, mask, extrinsic_matrix)
            # To verify
            # o3d.io.write_point_cloud(f"pcd_{seq_num}.ply", o3d_pcd)
            pcd_points.append(np.asarray(o3d_pcd.points))
            pcd_colors.append(np.asarray(o3d_pcd.colors))
            pcd_normals.append(np.asarray(o3d_pcd.normals))
        
        pcd = dict()
        pcd['points'] = pcd_points
        pcd['colors'] = pcd_colors
        pcd['normals'] = pcd_normals
        return pcd
    
    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """
        if key == 'actions':
            hd5key = "data/{}/{}/{}".format(ep, key, key)
        # In case we want to add eef pose to the action vector
        # if key == 'actions':
        #     actions = np.array(self.hdf5_file[f"data/{ep}/{key}/{key}"])
        #     ee_pos = np.array(self.hdf5_file[f"data/{ep}/proprioceptions/right_eef_pos"])[:-1]
        #     ee_quat = np.array(self.hdf5_file[f"data/{ep}/proprioceptions/right_eef_orn"])[:-1]
        #     ee_orn = R.from_quat(ee_quat).as_rotvec()
        #     ret = np.concatenate((actions, ee_pos, ee_orn), axis=1)
        #     # breakpoint()
        #     return ret
        elif key == 'obs/rgb':
            pass
        # TODO: Do the depth -> pcd transformation during data collection
        elif key == 'obs/pcd':
            return self.get_pcd(ep)            
        elif key == 'grasps':
            hd5key = "data/{}/extras/grasps".format(ep)
            ret = np.array(self.hdf5_file[hd5key])
            # ret = np.append(ret, ret[-1])
            return ret
        elif key == 'grasp_label':
            hd5key = "data/{}/extras/grasp_label".format(ep)
            ret = np.array(self.hdf5_file[hd5key])
            # ret = np.append(ret, ret[-1])
            return ret
        elif key == 'ft_label':
            hd5key = "data/{}/extras/ft_label".format(ep)
            ret = np.array(self.hdf5_file[hd5key])
            # ret = np.append(ret, ret[-1])
            return ret
        elif key == 'contacts':
            hd5key = "data/{}/extras/{}".format(ep, key)
            ret = np.array(self.hdf5_file[hd5key])
            # ret = np.append(ret, ret[-1])
            return ret
        elif key == 'object_dropped':
            hd5key = "data/{}/extras/object_dropped".format(ep)
            ret = np.array(self.hdf5_file[hd5key])
            # ret = np.append(ret, ret[-1])
            return ret

        ret = self.hdf5_file[hd5key]
        return ret
    
    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence (This is what gives us multiple sequences per episode [IMPORTANT])
        seq_begin_index = max(0, index_in_demo)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = 0
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length
        # print("seq_end_pad: ", seq_end_pad)

        # make sure we are not padding if specified.
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            # print("k: ", k)
            data = self.get_dataset_for_ep(demo_id, k)
            # seq[k] = data[seq_begin_index: seq_end_index].astype("float32")
            # Retain existing datatype
            if k == 'obs/pcd':
                seq['obs/pcd_points'] = np.array(data["points"][seq_begin_index: seq_end_index]) 
                seq['obs/pcd_colors'] = np.array(data["colors"][seq_begin_index: seq_end_index]) 
                seq['obs/pcd_normals'] = np.array(data["normals"][seq_begin_index: seq_end_index]) 
            elif k == 'contacts' or k =='grasps' or k == 'grasp_label' or k == "ft_label" or k == 'object_dropped':
                seq["labels"] = data[seq_begin_index+1: seq_end_index+1]
            else:
                seq[k] = data[seq_begin_index: seq_end_index]
            # change label from bool to float
            if k == 'grasps' or k == 'contacts' or k == 'grasp_label' or k == "ft_label" or k == 'object_dropped':
                seq["labels"] = seq["labels"].astype(float)

            # print("seq[k]: ", seq[k].shape)

        seq = pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        # if 'actions' not in keys:        
        #     print("seq: ", seq[k][-1, :])

        return seq
    
    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1, prefix="obs", obs_info_keys=None):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """
        obs = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            seq_length=seq_length,
        )
        
        
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix

        # prepare image observations from dataset
        # for k in obs:
            # uncomment later
            # obs[k] = obs[k][:, :, :, :3]
            # obs[k] = np.transpose(obs[k], (0, 3, 1, 2))

        return obs
    
    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences
    
    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        return self.get_item(index)
    
    def pc_normalize(self, pc, action):
        # breakpoint()
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m

        action[3:6] = (action[3:6] - centroid) / m

        return pc, action
    
    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        
        index_in_demo = index - demo_start_index 
        # print("index, index_in_demo: ", index, index_in_demo)

        # # end at offset index if not padding for seq length
        # demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        # end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length
        )
        # breakpoint()

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            seq_length=self.seq_length,
            prefix="obs",
            obs_info_keys=self.obs_info_keys
        )

        # Normalize pcd points and the actions
        # breakpoint()
        # meta["obs"]["pcd_points"][0], meta["actions"][0] = self.pc_normalize(meta["obs"]["pcd_points"][0], meta["actions"][0])
        # breakpoint()

        #TODO: commented out resize but bring it back
        # meta["obs"] = self.resize_image_observations(meta["obs"], max=255)

        return meta
    
def prepare_data(input_batch):
        # prepare_data is a custom collate function which not only batches the data from the dataset, but also
        # creates the output dictionaries which contain keys "video" and "actions"
        # print("----", type(input_batch), type(input_batch[0]))
        # print("input_batch: ", input_batch[0]['obs']['rgb'].shape, np.array(input_batch[0]['obs_info']).shape)
        xs = default_collate(input_batch)
        # return xs
        # xs['obs_info'] = np.array(xs['obs_info']).transpose(3, 0, 1, 2)

        # print("xs: ", xs["obs"]["pcd_points"].shape)
        # data_dict = {
        #         "points": [xs["obs"]["pcd_points"] for xs in input_batch],
        #         "actions": [xs["actions"] for xs in input_batch],
        #         "contacts": [xs["contacts"] for xs in input_batch]
        #     }
        data_dict = {
                "points": xs["obs"]["pcd_points"][:, 0],
                "actions": xs["actions"][:, 0], 
                # "contacts": xs["contacts"],
                # "grasps": xs["grasps"],
                "labels": xs["labels"]
            }
        
        return data_dict
        
        # print("xssssssss: ", xs['obs']['rgb'].shape, xs['obs_info'].shape)
        # print("ob_info_value: ", xs['obs_info'][0][0])
        # import matplotlib.pyplot as plt
        # img = xs['obs']['rgb'][0][0].permute(1,2,0)
        # seg_img = torch.argmax(img, axis=-1)
        # plt.imshow(seg_img)
        # plt.show()
        
        # Added by Arpit to resolve the shape mismatch error (32, 145) x (142, 128)
        # print("actions: ", xs['actions'][1:3])
        # print("----", xs['actions'][:, :, :3].shape, xs['actions'][:, :, -1:].shape)
        # xs['actions'] = torch.cat((xs['actions'][:, :, :3], xs['actions'][:, :, -1:]), dim=2)
        # print("xssssssss: ", xs['actions'].shape)
        
        # if only_state:
        #     data_dict = {
        #         "video": torch.cat(
        #             [
        #                 xs["obs"]["robot0_eef_pos"],
        #                 xs["obs"]["robot0_eef_quat"],
        #                 xs["obs"]["object"],
        #             ],
        #             dim=-1,
        #         ),
        #         "actions": xs["actions"],
        #     }
        # elif only_depth:
        #     # take depth video as the actual video
        #     data_dict = {
        #         "video": xs["obs"][f"{view}_depth"],
        #         "actions": xs["actions"],
        #     }
        # else:
        #     data_dict = {
        #         "video": xs["obs"][get_image_name(view)],
        #         "actions": xs["actions"],
        #     }
        # # data_dict["rewards"] = xs["rewards"]
        # if augmentation:
        #     data_dict["video"] = augmentation(data_dict["video"])

        # if f"{view}_seg" in xs["obs"]:
        #     data_dict["segmentation"] = xs["obs"][f"{view}_seg"]
        #     # from perceptual_metrics.mpc.utils import save_np_img
        #     # import ipdb; ipdb.set_trace()
        #     # for i in range(10):
        #     #     save_np_img(np.tile(((data_dict['segmentation'][0, i, 0] == i) * 60).cpu().numpy()[..., None], (1, 1, 3)).astype(np.uint8), f'seg_{i}')
        #     # zero out the parts of the segmentation which are not assigned label corresponding to object of interest
        #     # set the object label components to 1
        #     object_seg_indxs = [
        #         0,
        #         1,
        #         2,
        #         3,
        #     ]  # Seg index is 0 on the iGibson data, and 1 on Mujoco data
        #     arm_seg_indxs = [
        #         4,
        #         5,
        #         6,
        #     ]  # Seg index is 0 on the iGibson data, and 1 on Mujoco data
        #     seg_image = torch.zeros_like(data_dict["segmentation"])
        #     for object_seg_index in object_seg_indxs:
        #         seg_image[data_dict["segmentation"] == object_seg_index] = 1
        #     for arm_seg_index in arm_seg_indxs:
        #         seg_image[data_dict["segmentation"] == arm_seg_index] = 2
        #     not_either_mask = ~((seg_image == 1) | (seg_image == 2))
        #     seg_image[not_either_mask] = 0
        #     data_dict["segmentation"] = seg_image
        # else:
        #     data_dict["segmentation"] = None
        # if depth and not only_depth:
        #     data_dict["depth_video"] = xs["obs"][f"{view}_depth"]
        # if normal:
        #     data_dict["normal"] = xs["obs"][f"{view}_normal"]

        # # commented by Arpit
        # if "video" in data_dict:
        # #     # Normalize to [0, 1]
        #     data_dict["video"] = data_dict["video"] / 1.0
        # #     data_dict["video"] = data_dict["video"] / 255.0

        # # added by Arpit
        # xs["grasps"] = xs["grasps"].unsqueeze(2)
        # # print("xs_grasps.shape: ", type(xs['grasps']), xs['grasps'].shape)
        # data_dict['grasps'] = xs["grasps"]

        # # added by Arpit
        # # data_dict['obs_info'] = xs["obs_info"]

        # if postprocess_fn:
        #     data_dict = postprocess_fn(data_dict)

        # return data_dict