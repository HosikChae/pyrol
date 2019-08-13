from .Settings.RobotData import ALPHREDV3 as ALPHRED_V3
from gazebopy import RobotInterface as GazeboRobotInterface
from ALPHRED_V3_files.Util import MathFcn as MF
from ..gazebo_env import GazeboEnv
from .Settings.Macros_ALPHREDV3 import *
import numpy as np


NUM_JOINTS = 12
NUM_LEGS = 4


class ALPHREDV3(ALPHRED_V3, GazeboEnv):
    """
    Top View - o ~ end-effector position on robot
    actions: [Hip_Yaw_Leg1, # Positive rotates towards red axis positive rotation around z-axis (blue in gazebo)
              Hip_Pitch_Leg1,  # Positive rotates into ground
              Knee_Pitch_Leg1, # Positive rotates into ground
              Leg2, ..., Leg3, ..., Leg4]
                    o Leg4



    Leg1
    o                ________________o  Leg3 aligned with green axis in Gazebo
                    |
                    |
                    |
                    |
                    |
                    o  Leg2 aligned with red axis in Gazebo
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 mode='torque',
                 ctrl_cost_weight=0.5,  # 0.5
                 contact_cost_weight=5e-4,
                 healthy_reward=1.,  # 1.0
                 terminate_when_unhealthy=True,
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=1,
                 exclude_current_positions_from_observation=True,
                 simulation=True,
                 simulator_name='Gazebo',
                 step_size=0.007,  # 0.005,
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
            ctrl_range = np.array([[-1.] * NUM_JOINTS, [1.] * NUM_JOINTS], dtype=np.float64)
        else:
            raise ValueError('Please input proper simulation mode!')
        self.simulator.reset_simulation()

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_position_from_observation = exclude_current_positions_from_observation

        # self.z_rew = np.zeros(1)
        self.z_vel_rew = np.zeros(1)
        self.z_vel_old = np.zeros(1)
        # self.z_accl_rew = np.zeros(1)

        GazeboEnv.__init__(self, simulator=self.simulator, sim_dt=self.sim_dt, ctrl_range=ctrl_range)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    @property
    def is_healthy(self):
        state = self.state_vector().copy()
        z_limit = self.simulator.get_body_position().copy()[2] < 0.25

        q = self.simulator.get_current_position().copy()  # joints
        q = q.reshape((3, 4), order='F')
        q_limit = np.any(np.absolute(q[0]) > 0.6) or \
                  np.any(np.absolute(q[1]) > 1.57075) or \
                  np.any(q[2] < -0.3) or \
                  np.any(q[2] > KNEE_PITCH_LIMIT_PLUS)
        # q_limit = False

        f = self.simulator.get_current_force().copy()
        f_limit = np.any(np.absolute(f) > TORQUE_MAX)
        # f_limit = False

        quat = self.simulator.get_body_quaternion().copy()
        rot_xy = MF.quat2eul(quat)
        rot_limit = np.any(np.absolute(rot_xy) > 1.0472)  # 60 degrees in rad
        # rot_limit = False

        is_healthy = np.isfinite(state).all() and not z_limit and not q_limit and not f_limit and not rot_limit
        # if is_healthy is False:
        #     print('Alphred is not healthy!')
        # if z_limit:
        #     print('Z-Limit reached.')
        # if np.any(np.absolute(q[0]) > 0.6):
        #     print('Hip-Yaw limit reached.')
        # if np.any(np.absolute(q[1]) > 1.57075):
        #     print('Hip-Pitch reached.')
        # if np.any(q[2] < -0.3):
        #     print('Knee backwards limit.')
        # if np.any(q[2] > KNEE_PITCH_LIMIT_PLUS):
        #     print('Knee over-rotated.')

        return is_healthy

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        # raw_contact_forces = self.simulator.data.cfrc_ext
        # min_value, max_value = self._contact_force_range
        # contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        # TODO: Implement contact force measurements
        # contact_on = self.simulator.get_limb_contacts()  # TODO: Update gazebopy so can use
        # contact_on = np.ndarray((1,))
        # actuator_outputs = self.simulator.get_current_force()
        # contact_forces = np.sum(contact_on * actuator_outputs)
        contact_forces = 0.
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.square(np.sum(self.simulator.get_foot_contacts().copy()))
        return contact_cost

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def sim_step(self, action):
        self._take(action * TORQUE_MAX)  # because actions normalized to -1 to 1
        self.simulator.step_simulation()

    def jump_reward(self, z_before, z_after, action):
        quat = self.simulator.get_body_quaternion().copy()
        rot_xy = MF.quat2eul(quat)
        flat = np.sum(np.absolute(rot_xy))
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        total_cost = flat + ctrl_cost + contact_cost

        healthy_reward = self.healthy_reward
        z_vel = (z_after - z_before) / self.sim_dt

        if z_vel > self.z_vel_old and z_vel > 0.:
            self.z_vel_rew = np.exp((self.z_vel_rew + z_vel) * self.sim_dt * 10.)
        else:
            self.z_vel_rew = np.zeros(1)

        reward = healthy_reward + z_vel * 0.01 + self.z_vel_rew
        self.z_vel_old = z_vel

        return reward - total_cost

    def step(self, action):
        z_before = self.simulator.get_body_position()[2].copy()
        self.sim_step(action)
        z_after = self.simulator.get_body_position()[2].copy()
        # z_rel_vel = (z_after - z_before) / self.sim_dt
        # z_vel = self.simulator.get_body_velocity()[2].copy()
        # z_accl = self.simulator.get_imu_acceleration()[2].copy()
        #
        # upreward = np.exp(z_after - 0.5) if z_after > 0.5 else 0.
        #
        # self.z = np.absolute(self.z * z_after) + 0.1 if z_after > z_before else np.zeros(1)
        # self.z_vel = self.z_vel * np.absolute(z_vel) + 0.01 if self.z_vel * z_vel >= 0. else np.zeros(1)
        # self.z_accl = self.z_accl * np.absolute(z_accl/100.) + 0.01 \
        #     if self.z_accl * z_accl >= 0. else np.zeros(1)
        #
        # neg_reward = 0.0
        # if np.any(self.z_vel == 0) or np.any(self.z_accl == 0):
        #     neg_reward = -5.
        #
        # quat = self.simulator.get_body_quaternion().copy()
        # rot_xy = MF.quat2eul(quat)
        # flat = np.sum(np.absolute(rot_xy))
        #
        # jump_reward = upreward - flat + self.z * 0.5 + np.absolute(self.z_vel) * 1.5 + \
        #               np.absolute(self.z_accl) * 1.1 + neg_reward
        # TODO: Implement an accumalative reward for pos velocity and acceleration if stringed together

        # xy_position_before = self.get_body_com("torso")[:2].copy()
        # self.do_simulation(action, self.frame_skip)
        # xy_position_after = self.get_body_com("torso")[:2].copy()
        # xy_velocity = (xy_position_after - xy_position_before) / self.sim_dt
        # x_velocity, y_velocity = xy_velocity

        # ctrl_cost = self.control_cost(action)
        # contact_cost = self.contact_cost
        # healthy_reward = self.healthy_reward
        # rewards = jump_reward + healthy_reward
        # costs = ctrl_cost + contact_cost
        #
        # reward = rewards - costs  # TODO: Need to figure out weird bug: if get rid or np component in jump_reward broke

        reward = self.jump_reward(z_before=z_before, z_after=z_after, action=action)
        done = self.done
        observation = self._get_obs().T
        info = {
            # 'reward_forward': jump_reward,
            # 'reward_ctrl': -ctrl_cost,
            # 'reward_contact': -contact_cost,
            # 'reward_survive': healthy_reward,

            'z_position': z_after[0],

            'z_imu_velocity': self.z_vel_rew.copy(),
            # 'jump_reward': jump_reward,
        }
        # if done:
        #     print('Alphred died.')

        return observation, reward, done, info

    def _get_obs(self):
        # position = self.simulator.get_current_position().flat.copy()
        # velocity = self.simulator.get_current_velocity().flat.copy()
        # current_force = self.simulator.get_current_force().flat.copy()
        # contact_force = self.simulator.get_foot_contacts().flat.copy()  # np.array([self.contact_forces], dtype=np.float64)
        #
        # if self._exclude_current_position_from_observation:
        #     position = position[2:]
        #
        # observations = np.concatenate((position, velocity, current_force))

        return np.concatenate((self.state_vector().copy(), self.simulator.get_limb_contacts().flat.copy()))
        # return observations

    def reset_model(self):
        self.simulator.reset_simulation()
        self.reset_stand()
        # self.z = self.simulator.get_body_position()[2].copy()
        self.z_vel_rew = self.simulator.get_body_velocity()[2].copy()
        # self.z_accl = self.simulator.get_imu_acceleration()[2].copy()
        observation = self._get_obs()

        return observation

    def reset_stand(self):
        const = 1.0
        noise = self._reset_noise_scale
        hip_yaw = .0 * const
        hip_pitch = 1. * const
        knee_pitch = .01 * const
        scale_down = 12
        rescale = np.random.normal(loc=scale_down, scale=2 * noise, size=1)
        HY = np.random.normal(loc=hip_yaw, scale=.001 * noise, size=4)
        HP = np.random.normal(loc=hip_pitch, scale=.005 * noise, size=4)
        KP = np.random.normal(loc=knee_pitch, scale=.00001 * noise, size=4)
        action = np.concatenate((HY, HP, KP)).reshape((4, 3), order='F').ravel()

        stand = action * TORQUE_MAX
        for _ in range(6):
            self.sim_step(stand)

        stand = action / rescale
        for _ in range(15):
            self.sim_step(stand)

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

