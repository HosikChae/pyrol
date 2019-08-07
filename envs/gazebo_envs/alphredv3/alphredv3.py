from .Settings.RobotData import ALPHREDV3 as ALPHRED_V3
from .Settings.Macros_ALPHREDV3 import *
from gazebopy import RobotInterface as GazeboRobotInterface
from gazebo_envs.gazebo_env import GazeboEnv
import numpy as np


NUM_JOINTS = 12
NUM_LEGS = 4


class ALPHREDV3(ALPHRED_V3, GazeboEnv):
    metadata = {'render.modes': ['human']}

    # TODO: Check if need all this for initialization
    def __init__(self,
                 mode='torque',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=0.01,  # 1.0
                 terminate_when_unhealthy=True,
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 simulation=True,
                 simulator_name='Gazebo',
                 step_size=0.005,
                 name='ALPHRED'):
        ALPHRED_V3.__init__(self)
        self.name = name
        self.simulation = simulation
        self.simulator_name = simulator_name
        self.sim_dt = step_size
        self.init_sim()
        self.mode = mode
        if self.mode == 'pid':
            self.simulator.set_operating_mode(GazeboRobotInterface.POSITION_PID_MODE)
            self._take = self.simulator.set_command_positions
            ctrl_range = np.array([[HIP_YAW_LIMIT_MINUS, HIP_PITCH_LIMIT_MINUS, KNEE_PITCH_LIMIT_MINUS] * NUM_LEGS,
                                   [HIP_YAW_LIMIT_PLUS, HIP_PITCH_LIMIT_PLUS, KNEE_PITCH_LIMIT_PLUS] * NUM_LEGS])
        elif self.mode == 'torque':
            self.simulator.set_operating_mode(GazeboRobotInterface.TORQUE_MODE)
            self._take = self.simulator.set_command_force
            # TODO: Check if should rescale control range to -1 to 1
            ctrl_range = np.array([[-1] * NUM_JOINTS, [1] * NUM_JOINTS])
        else:
            raise ValueError('Please input proper simulation mode!')
        self.simulator.pause_physics()
        self.simulator.reset_simulation()

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_position_from_observation = exclude_current_positions_from_observation

        GazeboEnv.__init__(self, simulator=self.simulator, ctrl_range=ctrl_range)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def is_healthy(self):
        # TODO: Need to implement
        state = self.state_vector()
        a = self.simulator.get_body_quaternion()
        z_com = self.simulator.get_body_position()[2]
        c = self.simulator.get_current_position()

        is_healthy = (np.isfinite(state).all() and z_com >= 0.1786)
        return is_healthy

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(np.true_divide(action, TORQUE_MAX)))
        return control_cost

    @property
    def contact_forces(self):
        # raw_contact_forces = self.simulator.data.cfrc_ext
        # min_value, max_value = self._contact_force_range
        # contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        # TODO: Implement contact force measurements
        # contact_on = self.simulator.get_limb_contacts()  # TODO: Update gazebopy so can use
        contact_on = np.ndarray((1,))
        actuator_outputs = self.simulator.get_current_force()
        contact_forces = np.sum(contact_on * actuator_outputs)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.square(np.sum(self.simulator.get_foot_contacts()))
        return contact_cost

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xy_position_before = self.simulator.get_body_position()[:2].copy()
        # self.simulator.unpause_physics()
        self._take(np.double(action) * TORQUE_MAX)
        self.simulator.step_simulation()
        # self.simulator.pause_physics()
        xy_position_after = self.simulator.get_body_position()[:2].copy()

        # xy_position_before = self.get_body_com("torso")[:2].copy()
        # self.do_simulation(action, self.frame_skip)
        # xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.sim_dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity

        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        reward = reward[0]  # TODO: fix this workaround in baselines library
        # TODO: Implement done call on GazeboInterface
        done = False  # self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.simulator.get_current_position().flat.copy()
        velocity = self.simulator.get_current_velocity().flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_position_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    # TODO: Add a mode where there is noise added to the environment
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # TODO: Implement similar functionality for random initialization
        self.simulator.reset_simulation()
        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nq)
        # qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
        #     self.model.nv)
        # self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    # TODO: Gazebo camera setup to follow robot
    # DEFAULT_CAMERA_CONFIG = {
    #     'distance': 4.0,
    # }
    # def viewer_setup(self):
    #     for key, value in DEFAULT_CAMERA_CONFIG.items():
    #         if isinstance(value, np.ndarray):
    #             getattr(self.viewer.cam, key)[:] = value
    #         else:
    #             setattr(self.viewer.cam, key, value)

