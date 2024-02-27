# %%
"""
Quick documentationn on the available Programs:
control_states = [0,1,2,3,5,14,19,23,24,25,26,27];
(0):  Stop
(1):  TrackingCalibration
(2):  FTSCalibration
(3):  ConnectSpecimen
(5):  Return Home
(14): Handmove
(19): Flexion
(23): Pivot Shift
(24): Varus-Valgus Loading
(25): Internal-External Loading
(26): Lachmann Loading
(27): Lachmann Loading v2
"""
import os
from dm_control import mujoco
import json
import pystache
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from IPython.display import HTML
import numpy as np
import mujoco_viewer
from dm_control.rl.control import PhysicsError
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from config import EnvSettings, OverallSettings
from PIL import Image
import imageio
from pyvirtualdisplay import Display


#display = Display(visible=0, size=(640, 640))
#display.start()


TIME_LABEL = 'time in s'
SCENE = 'scene.xml'
LEGEND_LOC = 'lower right'


def get_data(smooth=True, t=0.8, name='data', path='measurement', filterit=False, filter_app=None, threshold=0.1, tste=None):
    with open(f'{path}/{name}.json') as f:
        data = json.load(f)
    if smooth:
        data = filter_force_torque_data(data, t=t)

    if filterit:
        data = filter_data(data, threshold=threshold)

    if filter_app is not None:
        data = filter_data_by_s_app(data, filter_app)

    if tste is not None:
        if len(tste) == 2:
            data = filter_data_by_ts_te(data, tste[0], tste[1])
        else:
            raise ValueError(
                "tste must be a single integer or a list of two integers")

    return data


def get_data_disk(name='data_disk', path='measurement'):
    with open(f'{path}/{name}.json') as f:
        data = json.load(f)
    return data


def get_data_hand(name='data_hand', path='measurement'):
    with open(f'{path}/{name}.json') as f:
        data = json.load(f)
    return data


def torque_lengths(torque_array):
    """Calculate the length of the torque vector for each time step"""
    if torque_array.ndim != 2 or torque_array.shape[1] != 3:
        raise ValueError(
            "Input array must have shape (n, 3) where n is the number of time steps.")

    return np.linalg.norm(torque_array, axis=1)


def moving_average(signal, window_size):
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def detect_oscillations(signal, threshold, min_duration=3, merge_range=300):
    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D numpy array.")

    diff_signal = np.diff(signal)
    oscillation_detected = np.abs(diff_signal) >= threshold

    for i in range(len(oscillation_detected) - min_duration + 1):
        if np.sum(oscillation_detected[i:i + min_duration]) == min_duration:
            oscillation_detected[i:i + min_duration] = True

    # Merge oscillation regions if the gap between them is smaller than or equal to merge_range
    i = 0
    while i < len(oscillation_detected) - merge_range:
        if oscillation_detected[i]:
            gap = 0
            while i + gap + 1 < len(oscillation_detected) and not oscillation_detected[i + gap + 1]:
                gap += 1
            if gap <= merge_range:
                oscillation_detected[i:i + gap + 1] = True
            i += gap
        i += 1

    my_arr = np.concatenate(([False], oscillation_detected))
    return np.logical_not(my_arr)


def normalized_cross_correlation(value, target):
    """
    Computes the Normalized Cross-Correlation (NCC) for multi-dimensional signals.

    :param value: A numpy array of shape (dim, tsteps), representing the first signal.
    :param target: A numpy array of shape (dim, tsteps), representing the second signal.
    :return: Sum of a numpy array of shape (dim,), where each element is the NCC for that dimension.
    """
    # Ensure that value and target are numpy arrays
    value = np.asarray(value)
    target = np.asarray(target)

    # Validate shapes
    if value.shape != target.shape:
        raise ValueError(
            "The shapes of 'value' and 'target' must be the same.")

    # Subtract the mean from each dimension
    value_mean_subtracted = value - value.mean(axis=1, keepdims=True)
    target_mean_subtracted = target - target.mean(axis=1, keepdims=True)

    # Calculate the numerator (cross-correlation term)
    numerator = np.sum(value_mean_subtracted * target_mean_subtracted, axis=1)

    # Calculate the denominator (product of standard deviations)
    denominator = np.sqrt(np.sum(value_mean_subtracted**2, axis=1)
                          * np.sum(target_mean_subtracted**2, axis=1))

    # Compute NCC for each dimension and handle the case where denominator is zero
    ncc = np.where(denominator != 0, numerator / denominator, 0)
    return ncc.sum()


def filter_data(data, threshold=0.1):
    """filter the data"""
    np_torque = np.array(data['torque_ext'])
    torque = torque_lengths(np_torque)
    osci_keep = detect_oscillations(torque, threshold)

    filter_data = {}
    for key, subdata in data.items():
        filter_data[key] = [t for t, flag in zip(subdata, osci_keep) if flag]

    return filter_data


def filter_data_by_s_app(data, values_to_keep):
    """Filter data in a dictionary by the values in the 's_app' key"""
    filtered_data = {}
    for key, value_list in data.items():
        if key == 's_app':
            filtered_data[key] = [
                val for val in value_list if val in values_to_keep]
        else:
            filtered_data[key] = [t for t, s_app in zip(
                value_list, data['s_app']) if s_app in values_to_keep]

    return filtered_data


def filter_data_by_ts_te(data, ts: int, te: int):
    filtered_data = {}
    for key, value_list in data.items():
        if len(value_list) < ts:
            return data
        filtered_data[key] = value_list[ts:te]
    return filtered_data


def find_between(s: str, first: str, last: str):
    """helper for string preformatting"""
    try:
        start = s.index(first) + len(first)
        start_pos = s.index(first)
        end = s.index(last, start)
        return s[start:end].replace('"', ''), start_pos, end
    except ValueError:
        return "", "", ""


def recursive_loading(xml_path, path_ext='../', st_s='<include file=', end_s='/>', template_mode=False):
    """recursively load subfiles"""

    with open(xml_path, "r") as stream:
        xml_string = stream.read()

    xml_string = xml_string.replace("./", path_ext)
    extra_file, start_p, end_p = find_between(xml_string, st_s, end_s)

    if template_mode:
        filename = extra_file.split('/')[-1].split('.')[0]
        extra_file = f'{path_ext}{filename}_template.xml'

    if len(extra_file) > 0:
        extra_string = recursive_loading(
            extra_file, path_ext=path_ext)

        spos = extra_string.index('<mujoco model=')
        end = extra_string.index('>', spos)
        extra_string = extra_string[:spos] + extra_string[end:]
        extra_string = extra_string.replace('</mujoco>', '')

        xml_string = xml_string[:start_p] + extra_string + xml_string[end_p:]

    return xml_string


def display_video(frames, framerate=60, dpi=600):
    height, width, _ = frames[0].shape
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())


def get_segment_data_knee(data, pos_start: int, pos_end: int):
    """get a data segment from the measurement"""
    seglen = pos_end - pos_start
    seg_dict = {}

    if len(data['femur_pos']) < pos_end:
        pos_end = len(data['femur_pos'])
        seglen = pos_end - pos_start

    seg_dict['fp'] = np.array(data['femur_pos'][pos_start:pos_end])
    seg_dict['fq'] = np.array(data['femur_quat'][pos_start:pos_end])
    seg_dict['tp'] = np.array(data['tibia_pos'][pos_start:pos_end])
    seg_dict['tq'] = np.array(data['tibia_quat'][pos_start:pos_end])

    seg_dict['ft'] = np.zeros((seglen, 6))
    seg_dict['ft'][:, :3] = np.array(data['force_ext'][pos_start:pos_end])
    seg_dict['ft'][:, 3:] = np.array(data['torque_ext'][pos_start:pos_end])

    return seg_dict


def get_segment_data_hand(data, pos_start: int, pos_end: int):
    """get a data segment from the measurement!!!"""
    seg_dict = {}
    seg_dict['ft'] = np.array(data['ctrl'][pos_start:pos_end])
    qpos = np.array(data['qpos_all']).T
    seg_dict['qpos_all'] = qpos[pos_start:pos_end]
    return seg_dict


def get_segment_data_disk(data, pos_start: int, pos_end: int):
    """get a data segment from the measurement"""
    seg_dict = {}
    if len(data['q']) < pos_end:
        pos_end = len(data['q'])

    seg_dict['q'] = np.array(data['q'][pos_start:pos_end])
    seg_dict['ft'] = np.array(data['ft'][pos_start:pos_end])

    return seg_dict


def filter_force_torque_data(data, t=0.8):
    """Filter the force and torque data in the given data dictionary using the given threshold t"""
    force = np.array(data['force_ext'])
    torque = np.array(data['torque_ext'])

    fx, fy, fz = map(lambda x: filter_array(x, t=t),
                     (force[:, 0], force[:, 1], force[:, 2]))
    mx, my, mz = map(lambda x: filter_array(x, t=t),
                     (torque[:, 0], torque[:, 1], torque[:, 2]))

    force = [(fxi, fyi, fzi) for fxi, fyi, fzi in zip(fx, fy, fz)]
    torque = [(mxi, myi, mzi) for mxi, myi, mzi in zip(mx, my, mz)]

    data['force_ext'] = force
    data['torque_ext'] = torque

    return data


def filter_array(arr, t=0.8, decimal_places=4):
    """Filter the array using the given threshold t and round the filtered values to the specified number of decimal places"""
    filtered_arr = []
    loc_value = arr[0]
    filtered_arr.append(round(loc_value, decimal_places))

    for value in arr[1:]:
        loc_value = t * loc_value + (1 - t) * value
        filtered_arr.append(round(loc_value, decimal_places))

    for i in range(len(arr)-1, 0, -1):
        loc_value = t * loc_value + (1 - t) * filtered_arr[i]
        filtered_arr[i] = round(loc_value, decimal_places)

    return filtered_arr


def get_error_knee(qpos, seg_data):
    err1 = ((seg_data['fq'] - qpos[:, 3:])**2).sum()
    err2 = ((seg_data['fp'] - qpos[:, :3])**2).sum()
    err = err1 + err2
    return err


def get_error_disk(qpos, seg_data):
    err = ((seg_data['q'] - qpos)**2).sum()
    return err


class MeasureDataset(Dataset):
    """Dataset that performs the simulation runs"""

    def __init__(self, cfg: OverallSettings, mode='train') -> None:
        """
        mode -> 'train' or 'valid'
        train_per -> percentage of data used for training
        randomness -> how much pertubation to use on the parameters
        seg_len -> length of a sequence
        """
        super(MeasureDataset, self).__init__()

        self.use_start_only = cfg.measure_dataset.use_start_only
        self.mode = mode
        self.train_per = cfg.measure_dataset.train_percentage / 100
        self.seg_len = cfg.env.max_episode_len
        self.shuffle = cfg.measure_dataset.shuffle
        self.data_file = cfg.env.data_path
        self.measure_path = cfg.measure_dataset.measure_path
        self.filter_osci = cfg.measure_dataset.filter_oscillations
        self.filter_apps = cfg.measure_dataset.filter_apps
        self.filter_tste = cfg.measure_dataset.filter_tste
        self.env_name = cfg.env.model_type
        self.start_pos = cfg.measure_dataset.start_pos

        if self.env_name == 'knee':
            self.get_segment_data = get_segment_data_knee

        if self.env_name == 'hand':
            self.get_segment_data = get_segment_data_hand

        if self.env_name == 'disk':
            self.get_segment_data = get_segment_data_disk

        # get the overall measurement data
        self.prepare_dataset()

        if self.use_start_only:
            self._getitem_method = self.get_start_only
        else:
            self._getitem_method = self.get_random_item

    def __getitem__(self, index):
        return self._getitem_method(index)

    def load_data(self):
        if self.env_name == 'knee':
            self.data = get_data(name=self.data_file, path=self.measure_path,
                                 filterit=self.filter_osci, filter_app=self.filter_apps,
                                 tste=self.filter_tste)

        if self.env_name == 'disk':
            self.data = get_data_disk(
                name=self.data_file, path=self.measure_path)

        if self.env_name == 'hand':
            self.data = get_data_hand(
                name=self.data_file, path=self.measure_path)

    def prepare_dataset(self):
        self.load_data()
        self.offset_pos = np.random.randint(self.seg_len)
        self.len_data = len(
            self.data[list(self.data.keys())[0]]) - self.offset_pos
        self.max_trainstep = int(round(self.len_data * self.train_per))
        self.val_len = self.len_data - self.max_trainstep
        self.min_valstep = self.max_trainstep + self.val_len % self.seg_len
        self.val_len = self.len_data - self.min_valstep
        self.max_trainstep -= self.max_trainstep % self.seg_len
        self.train_len = self.max_trainstep

        # select the relevant range of data
        if self.mode == 'train':
            self.len = int(round(self.train_len / self.seg_len))
            for key in self.data:
                self.data[key] = self.data[key][:self.max_trainstep][:]
        else:
            self.len = int(round(self.val_len / self.seg_len))
            for key in self.data:
                self.data[key] = self.data[key][self.min_valstep:][:]

        self.index_list = list(range(self.len))
        if self.shuffle:
            random.shuffle(self.index_list)

    def reshuffle(self):
        # only if we dont use the start
        if not self.use_start_only:
            self.prepare_dataset()

    def __len__(self):
        """Number of samples"""
        return self.len

    def get_start_only(self, index):
        return self.get_segment_data(self.data, self.start_pos, self.seg_len)

    def get_random_item(self, index):
        """perform all here"""
        idx = self.index_list[index]
        # get the positions
        start_p = idx * self.seg_len
        end_p = start_p + self.seg_len
        # get the data of one segment
        return self.get_segment_data(self.data, start_p, end_p)


class ParameterHandler:
    def __init__(self, parameters_file, param_path="../params"):
        self.parameters_file = f"{param_path}/{parameters_file}"

    def load_parameters(self, override=None):
        with open(self.parameters_file, "r") as stream:
            try:
                self.parameters = yaml.safe_load(stream)
                if override is not None:
                    self.parameters.update(override)
            except yaml.YAMLError as exc:
                print(exc)
                exit(0)


class ModelRenderer(ParameterHandler):
    def __init__(self, template_file, parameters_file, param_path="../params", path_ext="../"):
        super().__init__(parameters_file, param_path)
        self.template_file = template_file
        self.path_ext = path_ext
        self.render_template()

    def render_template(self, override=None):
        self.load_parameters(override=override)
        self.raw_model_str = recursive_loading(
            self.template_file, template_mode=True, path_ext=self.path_ext)
        self.update_model(self.parameters)

    def update_model(self, parameters):
        self.model = pystache.render(self.raw_model_str, parameters)
        self.physics = mujoco.Physics.from_xml_string(self.model)


class BaseModel(ModelRenderer):
    """the base model to inherit from"""

    def __init__(self, cfg: EnvSettings, dataset=None):
        """load model and raw xml string"""
        super().__init__(cfg.scene_path, cfg.param_file, cfg.param_path, cfg.path_ext)

        self.crit_speed = cfg.crit_speed
        self.freq = cfg.freq
        self.dt = 1 / cfg.freq
        self.done = False
        self.epoch_count = 0

        if dataset is not None:
            self.dset_active = True
            self.dset: MeasureDataset = dataset
            self.data = dataset.data
            self.new_epoch()
            self.get_new_data()

        else:
            self.dset_active = False
            self.data = get_data()

        self.t_step = 0
        self.mse = torch.nn.MSELoss()
        self.lam_reward = cfg.lam_reward
        self.mse_told = 0
        self.apply_ft_body_id = cfg.apply_ft_body_id
        self.pos_len = len(self.physics.data.qpos)
        self.vel_len = len(self.physics.data.qvel)

    def new_epoch(self):
        self.dset.reshuffle()
        self.iterset = iter(self.dset)
        self.episode_count = 0
        self.mse_told = 0

    def get_new_data(self):
        self.seg_data = next(self.iterset)
        self.episode_count += 1
        self.ft_ext = self.seg_data['ft']

        if self.episode_count >= self.dset.len - 1:
            self.new_epoch()
            self.epoch_count += 1

    def apply_force(self, ft_in):
        self.physics.data.xfrc_applied[self.apply_ft_body_id, :] = ft_in

    def step(self, ft_in=None):
        """apply one timestep in the simulation"""

        if ft_in is None:
            ft_in = self.ft_ext[self.t_step, :]

        tinit = self.physics.data.time
        self.t_step += 1

        if self.t_step >= self.max_sim_len:
            self.done = True

        while (self.physics.data.time - tinit) < self.dt:
            try:
                # integrator step for model simulation
                self.apply_force(ft_in)
                self.physics.step()
                cur_state_max = np.max(
                    np.abs(np.array(self.state())))

                # more stable: early braking method
                if cur_state_max > self.crit_speed:
                    self.done = True
                    return self.return_values()

            except PhysicsError:
                self.done = True
                return self.return_values()

        return self.return_values()

    def get_ft(self):
        return self.ft_ext[self.t_step, :]

    def simulate(self, ft_ext, freq=100, tend=1):
        """simulate for given time determined by length of ft_ext"""
        # init the trajectory
        sim_len = int(round(freq*tend))
        qpos = np.zeros((sim_len, self.pos_len))
        tpos = 0
        qpos[tpos, :] = self.physics.data.qpos

        self.apply_force(ft_ext[tpos, :])
        keep_simulating = True
        tinit = self.physics.data.time

        # simulate
        while tpos < sim_len - 1 and keep_simulating:
            # set the external force
            self.apply_force(ft_ext[tpos, :])

            # integrator step:
            self.physics.step()
            cur_state_max = np.max(
                np.abs(np.array(self.state())))

            # more stable: early braking method
            if cur_state_max > 3 * self.crit_speed:
                keep_simulating = False

            # sample by frequency
            if (self.physics.data.time - tinit) > self.dt:
                tpos += 1
                tinit = self.physics.data.time
                qpos[tpos, :] = self.physics.data.qpos

        return qpos

    def simulate_dq(self, ft_ext, freq=100, tend=1):
        """simulate for given time determined by length of ft_ext"""
        # init the trajectory
        sim_len = int(round(freq*tend))
        qpos = np.zeros((sim_len, self.vel_len))
        tpos = 0
        qpos[tpos, :] = self.physics.data.qvel

        self.apply_force(ft_ext[tpos, :])
        keep_simulating = True
        tinit = self.physics.data.time

        # simulate
        while tpos < sim_len - 1 and keep_simulating:
            # set the external force
            self.apply_force(ft_ext[tpos, :])

            # integrator step:
            self.physics.step()
            cur_state_max = np.max(
                np.abs(np.array(self.state())))

            # more stable: early braking method
            if cur_state_max > 3 * self.crit_speed:
                keep_simulating = False

            # sample by frequency
            if (self.physics.data.time - tinit) > self.dt:
                tpos += 1
                tinit = self.physics.data.time
                qpos[tpos, :] = self.physics.data.qvel

        return qpos

    def get_reward(self):
        """general reward function"""
        self.cur_pos = torch.tensor(self.pos())
        s_mea_t1 = self.get_measure_state()

        # the general distance between measurement and state
        mse_cur = -self.mse(s_mea_t1, self.cur_pos).detach().cpu().numpy()

        # Compute the Temporal Difference Error
        td_error = -abs(mse_cur - self.mse_told)

        # update the td error
        self.mse_told = mse_cur

        # reward for long simulations..?
        reward_len = 1.0

        # combine rewards
        reward = self.lam_reward * mse_cur + reward_len + \
            self.lam_reward * td_error
        return reward

    def pos(self):
        return self.physics.data.qpos

    def vel(self):
        pos = self.physics.data.qpos
        pos_len = len(pos)
        state = self.physics.state()
        vel = state[pos_len:]
        return vel

    def state(self):
        return self.physics.state()

    def complete_state(self, device='cpu'):
        state = torch.tensor(self.state())
        ft_ext = torch.tensor(self.ft_ext[self.t_step, :])
        complete_state = torch.cat([state, ft_ext], dim=0).to(device).float()
        return complete_state

    def return_values(self):
        """
        -> state, reward, done (dset active)
        -> state, pos, done (dset inactive)
        """
        self.check_max_time()

        if self.dset_active:
            return self.complete_state(), self.get_reward(), self.done

        return self.state(), self.pos(), self.done

    def check_max_time(self):
        """check if max time reached -> done is True"""
        if self.t_step >= (self.max_sim_len - 1):
            self.done = True

    def get_pixels(self):
        pixels = self.physics.render(height=1024, width=1024)
        return pixels

    def run(self, seg_data=None, parameters=None, freq=100):
        """interactively run the model"""

        # create the viewer object
        mjc_model = mujoco.MjModel.from_xml_string(self.model)
        data = mujoco.MjData(mjc_model)
        viewer = mujoco_viewer.MujocoViewer(mjc_model, data)
        tpos = 0
        tend = 100
        sim_len = int(round(freq*tend))

        if seg_data is not None:
            tend, ft = self.reset(
                seg_data=seg_data, parameters=parameters, freq=freq)
            sim_len = int(round(freq*tend))
            mjc_model = mujoco.MjModel.from_xml_string(self.model)
            data = mujoco.MjData(mjc_model)
            viewer = mujoco_viewer.MujocoViewer(mjc_model, data)

        # simulate and render
        while tpos < sim_len - 1:
            # apply external force if applied
            if seg_data is not None:
                self.apply_force(ft[tpos, :])
            # simulate forward
            mujoco.mj_step(mjc_model, data)
            viewer.render()

            if tpos < data.time * freq:
                tpos += 1

        # close
        viewer.close()

    def perform_all(self, parameters=None):
        t1 = 0
        t2 = t1 + 4000
        if parameters is None:
            parameters = self.parameters
        seg_data = get_segment_data_knee(self.data, t1, t2)
        return self.perform_run(seg_data, parameters=parameters, verbose=True, err=True)

    def perform_run(self, seg_data, freq=100, parameters=None, video=False, verbose=False, err=False):
        """perform a complete run by setting inital positions, applying ft and rec all data"""

        tend, ft = self.reset(
            seg_data=seg_data, parameters=parameters)

        # get the simulation state vector
        qpos = self.simulate(ft, freq=freq, tend=tend)
        loc_error = self.get_error(qpos, seg_data)

        if verbose:
            self.gernerate_video(qpos)
            if err:
                return self.plot_results(qpos, seg_data), loc_error
            return self.plot_results(qpos, seg_data)
        return qpos

    def gernerate_video(self, qpos):
        model = mujoco.MjModel.from_xml_string(self.model)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 1024, 1024)
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        frames = []

        # Loop through the timesteps, skipping every 10 steps
        for t in range(0, qpos.shape[0], 10):
            # Update the joint positions
            data.qpos[:qpos.shape[1]] = qpos[t, :]

            mujoco.mj_step(model, data)
            renderer.update_scene(
                data, scene_option=scene_option, camera="camera0")
            pixels = renderer.render()
            image = Image.fromarray((pixels).astype(np.uint8))
            frames.append(image)

        image_arrays = [np.array(img) for img in frames]
        self.epoch_count += 1
        with imageio.get_writer(f'./res/training_progress/{self.epoch_count}.mp4', mode='I', fps=20) as writer:
            for image_array in image_arrays:
                writer.append_data(image_array)

    def simulate_real_trajectory(self, seg_data, freq=100):
        frames = []
        len_traj = len(seg_data['fp'])
        for ind in range(len_traj):
            tend, ft = self.reset(seg_data=seg_data, t_sel=ind)
            frame = self.get_pixels()
            frames.append(frame)
        return display_video(frames, framerate=freq)


class DiskModel(BaseModel):
    """Contains the general strucutre for simulation models"""

    def __init__(self, cfg: EnvSettings, dataset=None):
        """load model and raw xml string"""
        super().__init__(cfg, dataset)

    def init_pos_and_model(self, seg_data, parameters, t_sel=0):
        # set start pos
        q0 = seg_data['q'][t_sel, :]

        # apply updated model
        if parameters is None:
            parameters = self.parameters

        # new only need to set the initial position in the qpos
        self.update_model(parameters)
        self.physics.data.qpos = q0

    def get_measure_state(self, device='cpu'):
        quat = torch.tensor(self.seg_data['tq'][self.t_step, :])
        s_t_mea = torch.zeros(self.pos_len, dtype=torch.float64).to(device)
        s_t_mea[:] = quat
        return s_t_mea

    def reset(self, seg_data, parameters=None, freq=100, t_sel=0):
        """initialize the model in the current state"""
        self.done = False
        self.t_step = 0
        self.init_pos_and_model(seg_data, parameters, t_sel)

        # get t_end
        tend = int(round((len(seg_data['q'][:, 0]) * 100) / freq)) / 100
        self.tend = tend
        self.max_sim_len = len(seg_data['q'][:, 0])

        # reset old mse
        self.mse_told = 0
        return tend, seg_data['ft']

    def reset_new(self, parameters, t_sel=0):
        self.get_new_data()
        self.done = False
        self.t_step = 0

        self.init_pos_and_model(self.seg_data, parameters, t_sel)
        tend = int(
            round((len(self.seg_data['q'][:, 0]) * 100) / self.freq)) / 100
        self.tend = tend
        self.max_sim_len = len(self.seg_data['q'][:, 0])

    def simulate_real_trajectory(self, seg_data, freq=100):
        frames = []
        len_traj = len(seg_data['q'])
        for ind in range(len_traj):
            _, _ = self.reset(seg_data=seg_data, t_sel=ind)
            frame = self.get_pixels()
            frames.append(frame)
        return display_video(frames, framerate=freq)

    def get_measure_state(self, device='cpu'):
        quat = torch.tensor(self.seg_data['q'][self.t_step, :])
        s_t_mea = torch.zeros(self.pos_len, dtype=torch.float64).to(device)
        s_t_mea[:] = quat
        return s_t_mea

    def perform_all(self, parameters=None):
        t1 = 0
        t2 = t1 + 1000
        if parameters is None:
            parameters = self.parameters
        seg_data = get_segment_data_disk(self.data, t1, t2)
        return self.perform_run(seg_data, parameters=parameters, verbose=True, err=True)

    def plot_results(self, qpos, seg_data):
        """Compare postion results"""
        font = {'size': 14}
        plt.rc('font', **font)

        len_t = len(seg_data['q'])
        time = np.linspace(0, len_t / 100, len_t)
        fig, ax = plt.subplots(1, 1, figsize=(6, 12), sharex=True)

        ax.plot(time, qpos)
        plt.gca().set_prop_cycle(None)
        ax.plot(time, seg_data['q'], '-.')
        ax.set_ylim([-0.5, 1])
        ax.grid()
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel('quaternion')
        plt.legend(['$q_w$', '$q_x$', '$q_y$', '$q_z$'], loc=LEGEND_LOC)

        fig.tight_layout()
        return fig

    def get_error(self, qpos, seg_data):
        err = ((seg_data['q'] - qpos)**2).sum()
        return err


class HandModel(BaseModel):
    """Contains the hand model with specifications"""

    def __init__(self, cfg: EnvSettings, dataset=None):
        """load model and raw xml string"""
        super().__init__(cfg, dataset)

    def init_pos_and_model(self, seg_data, parameters, t_sel=0):
        # set start pos
        q0 = seg_data['qpos_all'][t_sel, :]

        # apply updated model
        if parameters is None:
            parameters = self.parameters

        # new only need to set the initial position in the qpos
        self.update_model(parameters)
        self.physics.data.qpos = q0

    def apply_force(self, ft_in):
        """here the external activation is the control input"""
        self.physics.data.ctrl = ft_in

    def get_measure_state(self, device='cpu'):
        theta = torch.tensor(
            self.seg_data['qpos_all'][self.t_step, :], dtype=torch.float64)
        return theta.to(device)

    def get_error(self, qpos, seg_data):
        """error should be the mse between the simulated to the measured thetas"""
        err = normalized_cross_correlation(seg_data['qpos_all'], qpos)
        # err = ((seg_data['qpos_all'] - qpos)**2).sum()
        return err

    def reset(self, seg_data, parameters=None, freq=100, t_sel=0):
        """initialize the model in the current state"""
        self.done = False
        self.t_step = 0
        self.init_pos_and_model(seg_data, parameters, t_sel)

        # get t_end
        tend = int(round((len(seg_data['ft'][:, 0]) * 100) / freq)) / 100
        self.tend = tend
        self.max_sim_len = len(seg_data['ft'][:, 0])

        # reset old mse
        self.mse_told = 0
        return tend, seg_data['ft']

    def perform_all(self, parameters=None):
        t1 = 500
        t2 = t1 + 1000
        if parameters is None:
            parameters = self.parameters
        seg_data = get_segment_data_hand(self.data, t1, t2)
        return self.perform_run(seg_data, parameters=parameters, verbose=True, err=True)

    def plot_results(self, qpos, seg_data):
        """Compare postion results"""
        font = {'size': 14}
        plt.rc('font', **font)

        len_t = len(seg_data['qpos_all'])
        time = np.linspace(0, len_t / 100, len_t)
        fig, ax = plt.subplots(1, 1, figsize=(6, 12), sharex=True)

        ax.plot(time, qpos)
        plt.gca().set_prop_cycle(None)
        ax.plot(time, seg_data['qpos_all'], '-.')
        ax.set_ylim([-0.5, 1])
        ax.grid()
        ax.set_xlabel(TIME_LABEL)
        ax.set_ylabel('quaternion')
        plt.legend(['$Thumb_{CMC}$', '$Thumb_{MCP}$', '$Thumb_{PIP}$', '$Thumb_{DIP}$',
                   '$Index_{MCP}$', '$Index_{PIP}$', '$Index_{DIP}$'], loc=LEGEND_LOC)
        fig.tight_layout()
        return fig

    def reset_new(self, parameters, t_sel=0):
        self.get_new_data()
        self.done = False
        self.t_step = 0

        self.init_pos_and_model(self.seg_data, parameters, t_sel)
        tend = int(
            round((len(self.seg_data['ft'][:, 0]) * 100) / self.freq)) / 100
        self.tend = tend
        self.max_sim_len = len(self.seg_data['ft'][:, 0])


class KneeModel(BaseModel):
    """Contains the general knee model for simulation runs"""

    def __init__(self, cfg: EnvSettings, dataset=None):
        """load model and raw xml string"""
        super().__init__(cfg, dataset)

    def set_init_position(self, body_pos, body_quat, parameters, body_name='femur'):
        """update the body position"""
        find_str = f'<body name="{body_name}" '

        # get the position of body
        _, start_p, end_p = find_between(
            self.raw_model_str, find_str, '>')
        start_p = start_p + len(find_str)

        # also learn delta tibia-femur
        if body_name == 'tibia':
            body_pos[0] += float(parameters['tibia']['dx'])
            body_pos[1] += float(parameters['tibia']['dy'])
            body_pos[2] += float(parameters['tibia']['dz'])

            body_quat[0] += float(parameters['tibia']['dqw'])
            body_quat[1] += float(parameters['tibia']['dqx'])
            body_quat[2] += float(parameters['tibia']['dqy'])
            body_quat[3] += float(parameters['tibia']['dqz'])

        # prepare new string
        pos_str = f'pos="{body_pos[0]} {body_pos[1]} {body_pos[2]}"'
        quat_str = f'quat="{body_quat[0]} {body_quat[1]} {body_quat[2]} {body_quat[3]}"'
        replace_str = f'{pos_str} {quat_str}'

        # overwrite model str
        self.raw_model_str = self.raw_model_str[:start_p] + \
            replace_str + self.raw_model_str[end_p:]

    def get_measure_state(self, device='cpu'):
        pos = torch.tensor(self.seg_data['tp'][self.t_step, :])
        quat = torch.tensor(self.seg_data['tq'][self.t_step, :])
        s_t_mea = torch.zeros(self.pos_len, dtype=torch.float64).to(device)
        s_t_mea[:3] = pos
        s_t_mea[3:] = quat
        return s_t_mea

    def get_error(self, qpos, seg_data):
        err1 = ((seg_data['fq'] - qpos[:, 3:])**2).sum()
        err2 = ((seg_data['fp'] - qpos[:, :3])**2).sum()
        err = err1 + err2
        return err

    def init_pos_and_model(self, seg_data, parameters=None, t_sel=0):
        if parameters is None:
            parameters = self.parameters

        # set start pos
        fp0 = seg_data['fp'][t_sel, :]
        fq0 = seg_data['fq'][t_sel, :]
        self.set_init_position(fp0, fq0, parameters, body_name='femur')
        tp0 = seg_data['tp'][t_sel, :]
        tq0 = seg_data['tq'][t_sel, :]
        # self.set_init_position(tp0, tq0, parameters, body_name='tibia')

        # apply updated model
        self.update_model(parameters)

    def reset_new(self, parameters, t_sel=0):
        self.get_new_data()
        self.done = False
        self.t_step = 0

        self.init_pos_and_model(self.seg_data, parameters, t_sel)
        tend = int(
            round((len(self.seg_data['fp'][:, 0]) * 100) / self.freq)) / 100
        self.tend = tend
        self.max_sim_len = len(self.seg_data['fp'][:, 0])

    def reset(self, seg_data, parameters=None, freq=100, t_sel=0):
        """initialize the model in the current state"""
        self.done = False
        self.t_step = 0

        self.init_pos_and_model(seg_data, parameters, t_sel)

        # get t_end
        tend = int(round((len(seg_data['fp'][:, 0]) * 100) / freq)) / 100
        self.tend = tend
        self.max_sim_len = len(seg_data['fp'][:, 0])

        # reset old mse
        self.mse_told = 0

        return tend, seg_data['ft']

    # functions connected to visualization
    # -----------------------------------------------------------------------------------------------------
    def plot_results(self, qpos, seg_data):
        """Compare postion results"""
        font = {'size': 14}
        plt.rc('font', **font)

        len_t = len(seg_data['fp'])
        time = np.linspace(0, len_t / 100, len_t)
        fig, ax = plt.subplots(2, 1, figsize=(6, 12), sharex=True)

        plt.subplot(2, 1, 1)
        ax[0].plot(time, qpos[:, :3])
        plt.gca().set_prop_cycle(None)
        ax[0].plot(time, seg_data['fp'], '-.')

        ax[0].set_ylim([-1, 1])
        ax[0].grid()
        ax[0].set_ylabel('pos in m')
        plt.legend(['$x$', '$y$', '$z$'], loc=LEGEND_LOC)

        plt.subplot(2, 1, 2)
        ax[1].plot(time, qpos[:, 3:])
        plt.gca().set_prop_cycle(None)
        ax[1].plot(time, seg_data['fq'], '-.')
        ax[1].set_ylim([-0.5, 1])
        ax[1].grid()
        ax[1].set_xlabel(TIME_LABEL)
        ax[1].set_ylabel('quaternion')
        plt.legend(['$q_w$', '$q_x$', '$q_y$', '$q_z$'], loc=LEGEND_LOC)

        fig.tight_layout()
        return fig


def create_data_surr(lim1=0, lim2=5000):
    # load data
    data = get_data(smooth=True, name='data_new',
                    filterit=False, filter_app=[25], tste=[7190, 12990])

    # fill surrogate data
    data_surr = {}
    data_surr['q'] = data['femur_quat'][lim1:lim2]

    ft_arr = []
    for f, t in zip(data['force_ext'], data['torque_ext']):
        f, t = list(f), list(t)
        f.extend(t)
        ft_arr.append(f)
    data_surr['ft'] = ft_arr[lim1:lim2]

    with open('measurement/data_surr.json', 'w') as f:
        json.dump(data_surr, f)

    return data_surr


# %%
if __name__ == '__main__':
    data2 = get_data_disk('./data_new')

# %%


# %%
    cfg = EnvSettings(
        model_type='knee',
        path_ext='../',
        scene_path='../scene.xml',
        param_path='../params',
        param_file='parameters_init.yaml',
        crit_speed=20,
        max_episode_len=500,
        ft_input_len=6,
        freq=100,
        apply_ft_body_id=3,
        lam_reward=20,
    )

    t1 = 0000
    t2 = t1 + 5000
    data = get_data(smooth=True, name='data_new',
                    filterit=False, filter_app=[25])

    model = KneeModel(cfg)
    seg_data = get_segment_data_knee(data, t1, t2)
    fig, err2 = model.perform_run(
        seg_data, verbose=True, video=False, err=True)

    print(err2)

    # %%
    t1 = 7180
    t2 = t1 + 5000
    data = get_data(smooth=True, name='data_new',
                    filterit=False, filter_app=[25])
    # fig, err1 = model.perform_run(seg_data, verbose=True, video=False, err=True)

    data = get_data(smooth=True, name='data_new',
                    filterit=False, filter_app=[25], tste=[7190, 12990])
    # %%
    data_surr = create_data_surr()
    # %%
    plt.plot(data_surr['q'])
    # %%
    plt.plot(data_surr['ft'])
    # %%
