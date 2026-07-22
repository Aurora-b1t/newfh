import json
import os
import tempfile
import unittest

import numpy as np

from offline_replay import (
    environment_metadata,
    load_replay_into_buffer,
    save_replay_buffer,
)
from SAC import ReplayBuffer


def make_buffer(count=2, capacity=8):
    buffer = ReplayBuffer(capacity, num_heads=10, n_actions=20)
    for index in range(count):
        state = np.full((8, 8), index, dtype=np.float32)
        rewards = np.linspace(-1.0, 1.0, 10, dtype=np.float32) + index
        buffer.add(
            state,
            100.0,
            np.arange(10, dtype=np.int64),
            rewards,
            state + 1.0,
            110.0,
            index == count - 1,
        )
    return buffer


class ReplayBufferTests(unittest.TestCase):
    def test_one_add_stores_one_complete_step(self):
        buffer = make_buffer(count=1)
        sample = buffer.sample(1)

        self.assertEqual(1, buffer.size())
        self.assertEqual((1, 10), sample["actions"].shape)
        self.assertEqual((1, 10), sample["block_rewards"].shape)
        self.assertEqual((1,), sample["step_rewards"].shape)
        self.assertAlmostEqual(
            float(sample["step_rewards"][0]),
            float(sample["block_rewards"][0].mean()),
            places=6,
        )

    def test_rejects_invalid_action_and_reward_shapes(self):
        buffer = ReplayBuffer(4, num_heads=10, n_actions=20)
        state = np.zeros((8, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "actions must have shape"):
            buffer.add(state, 100, np.zeros(9), np.zeros(10), state, 100, False)
        with self.assertRaisesRegex(ValueError, "block_rewards must have shape"):
            buffer.add(state, 100, np.zeros(10), np.zeros(9), state, 100, False)
        with self.assertRaisesRegex(ValueError, "\[0, 19\]"):
            buffer.add(
                state,
                100,
                np.full(10, 20),
                np.zeros(10),
                state,
                100,
                False,
            )

    def test_rejects_observation_shape_drift_and_non_finite_values(self):
        buffer = ReplayBuffer(4, num_heads=10, n_actions=20)
        state = np.zeros((8, 8), dtype=np.float32)
        buffer.add(state, 100, np.zeros(10), np.zeros(10), state, 100, False)

        with self.assertRaisesRegex(ValueError, "differs from existing"):
            buffer.add(
                np.zeros((4, 4)),
                100,
                np.zeros(10),
                np.zeros(10),
                np.zeros((4, 4)),
                100,
                False,
            )
        invalid_state = state.copy()
        invalid_state[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            ReplayBuffer(4, 10, 20).add(
                invalid_state,
                100,
                np.zeros(10),
                np.zeros(10),
                invalid_state,
                100,
                False,
            )


class OfflineReplayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "replay_v3.npz")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_v3_round_trip(self):
        save_replay_buffer(self.path, make_buffer(), {"num_actions": 20})
        destination = ReplayBuffer(8, num_heads=10, n_actions=20)

        count, metadata = load_replay_into_buffer(
            self.path,
            destination,
            expected_observation_shape=(8, 8),
            expected_num_actions=20,
            expected_num_blocks=10,
        )

        self.assertEqual(2, count)
        self.assertEqual(3, metadata["format_version"])
        self.assertEqual(2, metadata["num_step_transitions"])
        self.assertEqual(2, destination.size())

    def test_rejects_v2(self):
        np.savez_compressed(
            self.path,
            metadata=np.asarray(json.dumps({"format_version": 2})),
        )
        with self.assertRaisesRegex(ValueError, "v1/v2"):
            load_replay_into_buffer(self.path, ReplayBuffer(8, 10, 20))

    def test_rejects_inconsistent_step_rewards(self):
        save_replay_buffer(self.path, make_buffer(), {"num_actions": 20})
        with np.load(self.path, allow_pickle=False) as archive:
            contents = {key: np.asarray(archive[key]) for key in archive.files}
        contents["step_rewards"] = contents["step_rewards"] + 1.0
        np.savez_compressed(self.path, **contents)

        with self.assertRaisesRegex(ValueError, "mean\(block_rewards\)"):
            load_replay_into_buffer(self.path, ReplayBuffer(8, 10, 20))

    def test_rejects_invalid_offline_action_range(self):
        save_replay_buffer(self.path, make_buffer(), {"num_actions": 20})
        with np.load(self.path, allow_pickle=False) as archive:
            contents = {key: np.asarray(archive[key]) for key in archive.files}
        contents["actions"] = contents["actions"].copy()
        contents["actions"][0, 0] = 20
        np.savez_compressed(self.path, **contents)

        with self.assertRaisesRegex(ValueError, "outside"):
            load_replay_into_buffer(
                self.path,
                ReplayBuffer(8, 10, 20),
                expected_num_actions=20,
            )

    def test_rejects_invalid_offline_head_shape(self):
        save_replay_buffer(self.path, make_buffer(), {"num_actions": 20})
        with np.load(self.path, allow_pickle=False) as archive:
            contents = {key: np.asarray(archive[key]) for key in archive.files}
        contents["actions"] = contents["actions"][:, :9]
        np.savez_compressed(self.path, **contents)

        with self.assertRaisesRegex(ValueError, "must have shape"):
            load_replay_into_buffer(self.path, ReplayBuffer(8, 10, 20))

    def test_strict_environment_metadata_rejects_before_buffer_write(self):
        save_replay_buffer(self.path, make_buffer(), {"num_actions": 20})
        destination = ReplayBuffer(8, 10, 20)
        current = environment_metadata(
            {"mode": "current"},
            {"jammer": "comb"},
            {"base_reward": 1.0},
        )

        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            load_replay_into_buffer(
                self.path,
                destination,
                expected_num_actions=20,
                current_environment_metadata=current,
                strict_environment_metadata=True,
            )
        self.assertEqual(0, destination.size())

    def test_strict_environment_metadata_accepts_matching_snapshot(self):
        current = environment_metadata(
            {"mode": "current"},
            {"jammer": "comb"},
            {"base_reward": 1.0},
        )
        metadata = {"num_actions": 20, **current}
        save_replay_buffer(self.path, make_buffer(), metadata)
        destination = ReplayBuffer(8, 10, 20)

        count, _metadata = load_replay_into_buffer(
            self.path,
            destination,
            expected_num_actions=20,
            current_environment_metadata=current,
            strict_environment_metadata=True,
        )
        self.assertEqual(2, count)
        self.assertEqual(2, destination.size())

    def test_non_strict_environment_metadata_allows_explicit_override(self):
        save_replay_buffer(self.path, make_buffer(), {"num_actions": 20})
        destination = ReplayBuffer(8, 10, 20)
        current = environment_metadata(
            {"mode": "current"},
            {"jammer": "comb"},
            {"base_reward": 1.0},
        )

        count, _metadata = load_replay_into_buffer(
            self.path,
            destination,
            expected_num_actions=20,
            current_environment_metadata=current,
            strict_environment_metadata=False,
        )
        self.assertEqual(2, count)
        self.assertEqual(2, destination.size())


if __name__ == "__main__":
    unittest.main()
