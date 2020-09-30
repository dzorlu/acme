# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recurrent DQfD (R2D3) agent implementation."""

import functools

from acme import datasets
from acme import specs
from acme import types as acme_types
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.r2d2 import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl

from typing import Optional

from acme import datasets

import coloredlogs, logging
coloredlogs.install(logging.INFO)
logger_ = logging.getLogger(__name__)

class R2D3(agent.Agent):
  """R2D3 Agent.

  This implements a single-process R2D2 agent that mixes demonstrations with
  actor experience.
  """

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               network: snt.RNNCore,
               target_network: snt.RNNCore,
               burn_in_length: int,
               trace_length: int,
               replay_period: int,
               demonstration_generator: iter,
               demonstration_ratio: float,
               model_directory: str,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               discount: float = 0.99,
               batch_size: int = 32,
               target_update_period: int = 100,
               importance_sampling_exponent: float = 0.2,
               epsilon: float = 0.01,
               learning_rate: float = 1e-3,
               log_to_bigtable: bool = False,
               log_name: str = 'agent',
               checkpoint: bool = True,
               min_replay_size: int = 1000,
               max_replay_size: int = 1000000,
               samples_per_insert: float = 32.0,
               ):

    extra_spec = {
        'core_state': network.initial_state(1),
    }
    # replay table
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(0.8),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders.SequenceAdder.signature(environment_spec,
                                                   extra_spec))
    # demonstation table.
    demonstration_table = reverb.Table(
        name='demonstration_table',
        sampler=reverb.selectors.Prioritized(0.8),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
        signature=adders.SequenceAdder.signature(environment_spec, extra_spec))

    # launch server
    self._server = reverb.Server([replay_table, demonstration_table], port=None)
    address = f'localhost:{self._server.port}'

    sequence_length = burn_in_length + trace_length + 1

    # Component to add things into replay and demo
    sequence_kwargs = dict(
        period=replay_period,
        sequence_length=sequence_length,
    )
    adder = adders.SequenceAdder(client=reverb.Client(address),
                                 **sequence_kwargs)
    priority_function = {demonstration_table.name: lambda x: 1.}
    demo_adder = adders.SequenceAdder(client=reverb.Client(address),
                                      priority_fns=priority_function,
                                      **sequence_kwargs)
    # play demonstrations and write
    # exhaust the generator
    # TODO: MAX REPLAY SIZE 
    _prev_action = 1 # this has to come from spec
    _add_first = True
    #include this to make datasets equivalent
    numpy_state = tf2_utils.to_numpy_squeeze( network.initial_state(1))
    for ts, action in demonstration_generator:
      if _add_first:
        demo_adder.add_first(ts)
        _add_first = False
      else:
        demo_adder.add(_prev_action, ts, extras=(numpy_state,))
      _prev_action = action
      # reset to new episode
      if ts.last():
        _prev_action = None
        _add_first = True

    # replay dataset
    max_in_flight_samples_per_worker = 2 * batch_size if batch_size else 100
    dataset = reverb.ReplayDataset.from_table_signature(
        server_address=address,
        table=adders.DEFAULT_PRIORITY_TABLE,
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
        num_workers_per_iterator=2, # memory perf improvment attempt  https://github.com/deepmind/acme/issues/33
        sequence_length=sequence_length,
        emit_timesteps=sequence_length is None)

    # demonstation dataset
    d_dataset = reverb.ReplayDataset.from_table_signature(
          server_address=address,
          table=demonstration_table.name,
          max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
          num_workers_per_iterator=2,
          sequence_length=sequence_length,
          emit_timesteps=sequence_length is None)

    dataset = tf.data.experimental.sample_from_datasets(
        [dataset, d_dataset],
        [1 - demonstration_ratio, demonstration_ratio])

    # Batch and prefetch.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    tf2_utils.create_variables(network, [environment_spec.observations])
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    learner = learning.R2D2Learner(
        environment_spec=environment_spec,
        network=network,
        target_network=target_network,
        burn_in_length=burn_in_length,
        dataset=dataset,
        reverb_client=reverb.TFClient(address),
        counter=counter,
        logger=logger,
        sequence_length=sequence_length,
        discount=discount,
        target_update_period=target_update_period,
        importance_sampling_exponent=importance_sampling_exponent,
        max_replay_size=max_replay_size,
        learning_rate=learning_rate,
        store_lstm_state=False,
    )

    self._checkpointer = tf2_savers.Checkpointer(
        directory=model_directory,
        subdirectory='r2d2_learner_v1',
        time_delta_minutes=15,
        objects_to_save=learner.state,
        enable_checkpointing=checkpoint,
    )

    self._snapshotter = tf2_savers.Snapshotter(
        objects_to_save=None, 
        time_delta_minutes=15000.,
        directory=model_directory)

    policy_network = snt.DeepRNN([
        network,
        lambda qs: trfl.epsilon_greedy(qs, epsilon=epsilon).sample(),
    ])

    actor = actors.RecurrentActor(policy_network, adder)
    observations_per_step = (float(replay_period * batch_size) /
                             samples_per_insert)
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=replay_period * max(batch_size, min_replay_size),
        observations_per_step=observations_per_step)

  def update(self):
    updated = super().update()
    if updated:
      self._snapshotter.save()
      self._checkpointer.save()


def _sequence_from_episode(observations: acme_types.NestedTensor,
                           actions: tf.Tensor,
                           rewards: tf.Tensor,
                           discounts: tf.Tensor,
                           extra_spec: acme_types.NestedSpec,
                           period: int,
                           sequence_length: int):
  """Produce Reverb-like sequence from a full episode.

  Observations, actions, rewards and discounts have the same length. This
  function will ignore the first reward and discount and the last action.

  This function generates fake (all-zero) extras.

  See docs for reverb.SequenceAdder() for more details.

  Args:
    observations: [L, ...] Tensor.
    actions: [L, ...] Tensor.
    rewards: [L] Tensor.
    discounts: [L] Tensor.
    extra_spec: A possibly nested structure of specs for extras. This function
      will generate fake (all-zero) extras.
    period: The period with which we add sequences.
    sequence_length: The fixed length of sequences we wish to add.

  Returns:
    (o_t, a_t, r_t, d_t, e_t) Tuple.
  """

  length = tf.shape(rewards)[0]
  first = tf.random.uniform(shape=(), minval=0, maxval=length, dtype=tf.int32)
  first = first // period * period  # Get a multiple of `period`.
  to = tf.minimum(first + sequence_length, length)

  def _slice_and_pad(x):
    pad_length = sequence_length + first - to
    padding_shape = tf.concat([[pad_length], tf.shape(x)[1:]], axis=0)
    result = tf.concat([x[first:to], tf.zeros(padding_shape, x.dtype)], axis=0)
    result.set_shape([sequence_length] + x.shape.as_list()[1:])
    return result

  o_t = tree.map_structure(_slice_and_pad, observations)
  a_t = tree.map_structure(_slice_and_pad, actions)
  r_t = _slice_and_pad(rewards)
  d_t = _slice_and_pad(discounts)
  start_of_episode = tf.equal(first, 0)
  start_of_episode = tf.expand_dims(start_of_episode, axis=0)
  start_of_episode = tf.tile(start_of_episode, [sequence_length])

  def _sequence_zeros(spec):
    return tf.zeros([sequence_length] + spec.shape, spec.dtype)

  e_t = tree.map_structure(_sequence_zeros, extra_spec)

  key = tf.zeros([sequence_length], tf.uint64)
  probability = tf.ones([sequence_length], tf.float64)
  table_size = tf.ones([sequence_length], tf.int64)
  priority = tf.ones([sequence_length], tf.float64)
  info = reverb.SampleInfo(
      key=key,
      probability=probability,
      table_size=table_size,
      priority=priority)
  return reverb.ReplaySample(
      info=info,
      data=adders.Step(
          observation=o_t,
          action=a_t,
          reward=r_t,
          discount=d_t,
          start_of_episode=start_of_episode,
          extras=e_t))
