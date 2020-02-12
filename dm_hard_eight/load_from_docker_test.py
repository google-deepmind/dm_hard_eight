# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for dm_hard_eight.load_from_docker."""

from absl import flags
from absl.testing import absltest
from dm_env import test_utils
import dm_hard_eight

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'docker_image_name', None,
    'Name of the Docker image that contains the Hard Eight Tasks. '
    'If None, uses the default dm_hard_eight name')


class LoadFromDockerTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self, level_name='ball_room_navigation_cubes'):
    return dm_hard_eight.load_from_docker(
        name=FLAGS.docker_image_name,
        settings=dm_hard_eight.EnvironmentSettings(
            seed=123, level_name=level_name))

  def test_action_spec(self):
    action_spec = set(self.environment.action_spec().keys())
    expected_actions = {
        'STRAFE_LEFT_RIGHT', 'MOVE_BACK_FORWARD', 'LOOK_LEFT_RIGHT',
        'LOOK_DOWN_UP', 'HAND_ROTATE_AROUND_RIGHT', 'HAND_ROTATE_AROUND_UP',
        'HAND_ROTATE_AROUND_FORWARD', 'HAND_PUSH_PULL', 'HAND_GRIP'
    }
    self.assertSetEqual(expected_actions, action_spec)


if __name__ == '__main__':
  absltest.main()
