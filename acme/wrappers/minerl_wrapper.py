"""A wrapper that puts the previous action, reward, and
  observation vector into observation
"""
from sklearn.cluster import MiniBatchKMeans, KMeans
from typing import NamedTuple

from acme import types
from acme.wrappers import base
from acme.utils import loggers
from acme import specs

import dm_env
import tree
import tqdm
from collections import OrderedDict

import minerl
import numpy as np
import pickle
import os

import coloredlogs, logging
coloredlogs.install(logging.INFO)
logger = logging.getLogger(__name__)

class OVAR(NamedTuple):
  """Container for (Observation, ObsVector, Action, Reward) tuples."""
  observation: types.Nest
  obs_vector: types.Nest
  action: types.Nest
  reward: types.Nest


class MineRLWrapper(base.EnvironmentWrapper):
  """MineRL wrapper."""

  def __init__(self, 
               environment: dm_env.Environment, 
               num_actions: int,
               k_means_path: str,
               dat_loader: minerl.data.data_pipeline.DataPipeline,
               train: bool = True,
               num_samples: int = 1000
               ):
    super().__init__(environment)
    self.num_actions = num_actions

    self._prev_action: types.NestedArray
    self._prev_reward: types.NestedArray
    self.num_actions = num_actions

    file_path = os.path.join(k_means_path, 'k_means.pkl')

    if train:
      self.k_means = KMeans(n_clusters=num_actions, random_state=0)

      # replay trajectories
      trajectories = dat_loader.get_trajectory_names()[:5]
      actions = list()
      for t, trajectory in enumerate(trajectories):
        logger.info({str(t): trajectory})
        for i, (state, a, r, _, done, meta) in enumerate(dat_loader.load_data(trajectory, include_metadata=True)):    
          action = a['vector'].reshape(1, 64)
          actions.append(action)
      actions = np.vstack(actions)
      self.k_means.fit(actions)
      logger.info({'finished': len(actions)})
      del actions
      pickle.dump(self.k_means, open(file_path, 'wb'))
      logger.info({'persisted k-means under': file_path})
    else:
      self.k_means = pickle.load(open(file_path,'rb'))
      logger.info({'loaded k-means from': file_path})


  def reset(self) -> dm_env.TimeStep:
    # Initialize with zeros of the appropriate shape/dtype.
    self._prev_action = tree.map_structure(
        lambda x: x.generate_value(), self.action_spec())
    self._prev_reward = tree.map_structure(
        lambda x: x.generate_value(), self._environment.reward_spec())
    timestep = self._environment.reset()
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def step(self, action) -> dm_env.TimeStep:
    cont_action = self.map_action(action)
    timestep = self._environment.step(cont_action)
    new_timestep = self._augment_observation(timestep)
    self._prev_action = action #save discrete action
    self._prev_reward = timestep.reward
    return new_timestep

  def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    ovar = OVAR(observation=timestep.observation['pov'].astype(np.float32), #env output uint8
                obs_vector=timestep.observation['vector'],
                action=self._prev_action,
                reward=self._prev_reward)
    return timestep._replace(observation=ovar)

  def map_action(self, action: types.NestedArray) -> types.NestedArray:
    # map to cont action space that env demands
    return OrderedDict({'vector': self.k_means.cluster_centers_[action]})

  def observation_spec(self):
    # pov obs are uint8, but required type is float.
    obs_spec = self._observation_spec['pov']
    obs_spec = specs.BoundedArray(shape=obs_spec.shape,
      dtype=np.float32,
      minimum=obs_spec.minimum,
      maximum=obs_spec.maximum,
      name=obs_spec.name)

    return OVAR(observation=obs_spec,
                obs_vector=self._observation_spec['vector'],
                action=self.action_spec(),
                reward=self.reward_spec())

  def action_spec(self):
    return specs.DiscreteArray(num_values=self.num_actions)


