import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Replay:
    """Simple inspectable one-game HDF5 replay store.

    Transition-level datasets:
        obs:         (N, H, W, C)
        action:      (N,)
        reward:      (N,)
        done:        (N,)
        episode_idx: (N,)
        step_idx:    (N,)

    Episode-level datasets:
        ep_len:      (num_episodes,)
        ep_offset:   (num_episodes,)

    mode='w' so that each time you run the train, the file is re-written, if 'a', its only append
    """

    def __init__(
        self,
        path,
        obs_shape=(3, 224, 224),
        mode="w",
        metadata=None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.obs_shape = tuple(obs_shape)
        self.file = h5py.File(self.path, mode)

        if mode != "r":
            self._init_datasets(metadata)
            self._restore_episode_state()

    def _init_datasets(self, metadata=None):
        if "obs" in self.file:
            return

        metadata = metadata or {}
        maxshape = (None,)

        self.file.create_dataset(
            "obs",
            shape=(0, *self.obs_shape),
            maxshape=(None, *self.obs_shape),
            dtype="float32",
            chunks=(1, *self.obs_shape),
        )

        self.file.create_dataset(
            "action",
            shape=(0,),
            maxshape=maxshape,
            dtype="int64",
            chunks=True,
        )

        self.file.create_dataset(
            "reward",
            shape=(0,),
            maxshape=maxshape,
            dtype="float32",
            chunks=True,
        )

        self.file.create_dataset(
            "done",
            shape=(0,),
            maxshape=maxshape,
            dtype="int64",
            chunks=True,
        )

        self.file.create_dataset(
            "episode_idx",
            shape=(0,),
            maxshape=maxshape,
            dtype="int64",
            chunks=True,
        )

        self.file.create_dataset(
            "step_idx",
            shape=(0,),
            maxshape=maxshape,
            dtype="int64",
            chunks=True,
        )

        self.file.create_dataset(
            "ep_len",
            shape=(0,),
            maxshape=maxshape,
            dtype="int32",
            chunks=True,
        )

        self.file.create_dataset(
            "ep_offset",
            shape=(0,),
            maxshape=maxshape,
            dtype="int64",
            chunks=True,
        )

        self.file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)

    def _restore_episode_state(self):
        """Restore episode tracking when opening an existing replay file in append mode."""

        if self.size == 0:
            self._current_episode_idx = None
            self._need_new_episode = True
            return

        if self.num_episodes == 0:
            raise RuntimeError(
                "Replay file has transitions but no episode metadata. "
                "The file may be corrupted or from an older replay format."
            )

        last_done = bool(self.file["done"][-1])

        if last_done:
            self._current_episode_idx = None
            self._need_new_episode = True
        else:
            self._current_episode_idx = self.num_episodes - 1
            self._need_new_episode = False

    @property
    def size(self):
        return int(self.file["action"].shape[0])

    @property
    def num_episodes(self):
        return int(self.file["ep_len"].shape[0])

    def _start_new_episode(self, offset):
        episode_idx = self.num_episodes

        self.file["ep_len"].resize((episode_idx + 1,))
        self.file["ep_offset"].resize((episode_idx + 1,))

        self.file["ep_len"][episode_idx] = 0
        self.file["ep_offset"][episode_idx] = int(offset)

        self._current_episode_idx = episode_idx
        self._need_new_episode = False

    def append(
        self,
        obs,
        action,
        reward,
        done,
    ):
        idx = self.size

        if self._need_new_episode:
            self._start_new_episode(offset=idx)

        episode_idx = self._current_episode_idx
        step_idx = int(self.file["ep_len"][episode_idx])

        for name in (
            "obs",
            "action",
            "reward",
            "done",
            "episode_idx",
            "step_idx",
        ):
            self.file[name].resize((idx + 1, *self.file[name].shape[1:]))

        self.file["obs"][idx] = np.asarray(obs, dtype=np.float32)
        self.file["action"][idx] = int(action)
        self.file["reward"][idx] = float(reward)
        self.file["done"][idx] = int(done)

        self.file["episode_idx"][idx] = int(episode_idx)
        self.file["step_idx"][idx] = int(step_idx)

        self.file["ep_len"][episode_idx] = step_idx + 1

        if done:
            self._current_episode_idx = None
            self._need_new_episode = True

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ReplaySequenceDataset(Dataset):
    """Read-only Dataset that returns fixed-length replay sequences.

    Each item contains, by default:

        obs:    [T, C, H, W]
        action: [T]
        reward: [T]
        done:   [T]

    The replay file may also contain:
        episode_idx: [T]
        step_idx:    [T]

    If allow_cross_episode=False (default), ep_offset and ep_len are
        used to avoid sequences that cross episode boundaries.

    This dataset supports refresh() so it can be reused while the underlying
    HDF5 replay file grows between training phases.

    Important:
        - Call refresh() only when no DataLoader workers are actively reading.
        - After refresh(), recreate the DataLoader if num_workers > 0.
        - The replay writer should call flush() before refresh().
    """

    def __init__(
        self,
        path,
        seq_len: int,
        batch_size: int,
        num_batches: int,
        allow_cross_episode: bool = False,
        keys: Sequence[str] = ("obs", "action", "reward", "done"),
    ):
        self.path = Path(path)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.keys = tuple(keys)
        self.allow_cross_episode = allow_cross_episode

        self._file = None

        self.num_steps = 0
        self.num_episodes = 0
        self.metadata = {}
        self.valid_starts = np.array([], dtype=np.int64)
        self.chosen_starts = np.array([], dtype=np.int64)

        self.refresh()

    def refresh(self):
        """Refresh dataset metadata and rebuild valid sequence starts.

        This should be called after the replay writer has appended new data and
        flushed the HDF5 file.

        This method closes any currently open read handle, reopens the file
        briefly, reloads the replay length and episode metadata, rebuilds
        self.valid_starts, and samples self.chosen_starts.
        """
        self.close()

        with h5py.File(self.path, "r") as f:
            self.num_steps = int(f["action"].shape[0])
            self.num_episodes = int(f["ep_len"].shape[0])

            metadata_json = f.attrs.get("metadata_json", "{}")
            self.metadata = json.loads(metadata_json)

            self.valid_starts = self._build_valid_starts(f)
            self.chosen_starts = self._build_chosen_starts()

        return len(self.chosen_starts)

    def _build_valid_starts(self, f):
        if self.num_steps < self.seq_len:
            valid_starts = np.array([], dtype=np.int64)

        elif self.allow_cross_episode:
            valid_starts = np.arange(
                0,
                self.num_steps - self.seq_len + 1,
                dtype=np.int64,
            )

        else:
            valid_starts = []

            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]

            for offset, length in zip(ep_offset, ep_len):
                offset = int(offset)
                length = int(length)

                if length < self.seq_len:
                    continue

                for start in range(offset, offset + length - self.seq_len + 1):
                    valid_starts.append(start)

            valid_starts = np.asarray(valid_starts, dtype=np.int64)

        return valid_starts

    def _build_chosen_starts(self):
        num_chosen = self.batch_size * self.num_batches

        if len(self.valid_starts) == 0:
            return np.array([], dtype=np.int64)

        num_second_half = int(np.ceil(num_chosen * 0.7))
        num_anywhere = num_chosen - num_second_half

        second_half_starts = self.valid_starts[self.valid_starts >= self.num_steps // 2]
        if len(second_half_starts) == 0:
            second_half_starts = self.valid_starts

        chosen_starts = np.concatenate(
            [
                np.random.choice(
                    second_half_starts, size=num_second_half, replace=True
                ),
                np.random.choice(self.valid_starts, size=num_anywhere, replace=True),
            ]
        )
        np.random.shuffle(chosen_starts)

        return chosen_starts.astype(np.int64)

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        return len(self.chosen_starts)

    def __getitem__(self, idx):
        start = int(self.chosen_starts[idx])
        end = start + self.seq_len

        f = self._ensure_open()
        sample = {}

        for key in self.keys:
            value = f[key][start:end]

            if key == "obs":
                tensor = torch.from_numpy(np.asarray(value)).float()

            elif key in {"action", "done", "episode_idx", "step_idx"}:
                tensor = torch.from_numpy(np.asarray(value)).long()

            elif key == "reward":
                tensor = torch.from_numpy(np.asarray(value)).float()

            else:
                tensor = torch.from_numpy(np.asarray(value))

            sample[key] = tensor

        return sample

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        self.close()
