from collections import OrderedDict
import os


from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import gazebopy
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))  # TODO: Check if can leave as is
        high = np.full(observation.shape, float('inf'))  # TODO: Check if can leave as is
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments"""

    def __init__(self, simulator, sim_dt, ctrl_range, frame_skip=None, rgb_rendering_tracking=False):
        self.frame_skip = frame_skip
        # self.data = self.simulator.data
        self.viewer = None
        self.rgb_rendering_tracking = rgb_rendering_tracking
        self._viewers = {}

        self.simulator = simulator
        self.sim_dt = sim_dt

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.sim_dt))
        }

        self.init_qpos = self.simulator.get_current_position().ravel().copy()
        self.init_qvel = self.simulator.get_current_velocity().ravel().copy()

        self._set_action_space(ctrl_range)

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self, ctrl_range):
        # bounds = self.model.actuator_ctrlrange.copy()
        bounds = ctrl_range.copy()
        low, high = bounds
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        # self.simulator.reset_simulation()
        ob = self.reset_model()  # TODO: Implement observation state and box limits
        return ob

    # def set_state(self, qpos, qvel):
    #     assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
    #     assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
    #     old_state = self.simulator.get_state()
    #     new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
    #                                      old_state.act, old_state.udd_state)
    #     self.simulator.set_state(new_state)
    #     self.simulator.forward()

    # Should be taken care by RobotData.py
    # @property
    # def dt(self):
    #     # return self.model.opt.timestep * self.frame_skip
    #     #  TODO: Need interface to pull dt from Gazebo
    #     return 0.0001  # TODO: Check what this actually is, right now placeholder

    # TODO: Check if needed
    # def do_simulation(self, ctrl, n_frames):
    #     self.simulator.data.ctrl[:] = ctrl
    #     for _ in range(n_frames):
    #         self.simulator.step_simulation()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # if mode == 'rgb_array':
        #     camera_id = None
        #     camera_name = 'track'
        #     if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
        #         camera_id = self.model.camera_name2id(camera_name)
        #     self._get_viewer(mode).render(width, height, camera_id=camera_id)
        #     # window size used for old mujoco-py:
        #     data = self._get_viewer(mode).read_pixels(width, height, depth=False)
        #     # original image is upside-down, so flip it
        #     return data[::-1, :, :]
        # elif mode == 'depth_array':
        #     self._get_viewer(mode).render(width, height)
        #     # window size used for old mujoco-py:
        #     # Extract depth part of the read_pixels() tuple
        #     data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
        #     # original image is upside-down, so flip it
        #     return data[::-1, :]
        # elif mode == 'human':
        #     self._get_viewer(mode).render()
        # TODO: Implement Gazebo interface to render in optimal window setting
        pass

    def close(self):
        # if self.viewer is not None:
        #     # self.viewer.finish()
        #     self.viewer = None
        #     self._viewers = {}
        # TODO: Implement
        pass

    # TODO: Check if actually need
    # def _get_viewer(self, mode):
    #     self.viewer = self._viewers.get(mode)
    #     if self.viewer is None:
    #         if mode == 'human':
    #             self.viewer = mujoco_py.MjViewer(self.simulator)
    #         elif mode == 'rgb_array' or mode == 'depth_array':
    #             self.viewer = mujoco_py.MjRenderContextOffscreen(self.simulator, -1)
    #
    #         self.viewer_setup()
    #         self._viewers[mode] = self.viewer
    #     return self.viewer

    # TODO: Check if needed
    # def get_body_com(self, body_name):
    #     return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.simulator.get_body_position().flat,  # Body
            self.simulator.get_body_quaternion().flat,
            self.simulator.get_body_velocity().flat,
            self.simulator.get_imu_acceleration().flat,
            self.simulator.get_imu_angular_rate().flat,
            self.simulator.get_current_position().flat,  # Joints
            self.simulator.get_current_velocity().flat,
            self.simulator.get_current_force().flat,
            # self.simulator.get_limb_contacts().flat
        ])

        # return np.concatenate([
        #     self.simulator.get_current_position().flat,
        #     self.simulator.get_current_velocity().flat
        # ])




