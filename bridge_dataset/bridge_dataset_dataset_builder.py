from typing import Iterator, Tuple, Any
import traceback
import glob
import os
import numpy as np
import random
from datetime import datetime
import pickle
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from bridge_dataset.conversion_utils import MultiThreadedDatasetBuilder

# we ignore the small amount of data that contains >4 views
N_VIEWS = 4
IMAGE_SIZE = (480, 640)
INPUT_PATH = "/nfs/kun2/users/homer/datasets/bridge_data_all/raw"
DEPTH = 5
TRAIN_PROPORTION = 0.9


def read_resize_image(path: str, size: Tuple[int, int]) -> np.array:
    """Reads, decodes, resizes an image."""
    data = tf.io.read_file(path)
    image = tf.image.decode_image(data)
    image = tf.image.resize(image, size, method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    orig_names = [f"images{i}" for i in range(N_VIEWS)]
    new_names = [f"image_{i}" for i in range(N_VIEWS)]

    def _parse_examples(episode_path, camera_topics):
        # load raw data

        def process_images(path):  # processes images at a trajectory level
            image_dirs = set(os.listdir(str(path))).intersection(set(orig_names))
            image_paths = [
                sorted(
                    glob.glob(os.path.join(path, image_dir, "im_*.jpg")),
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                for image_dir in image_dirs
            ]

            filenames = [[path.split("/")[-1] for path in x] for x in image_paths]
            assert all(x == filenames[0] for x in filenames)

            d = {
                image_dir: [read_resize_image(path, IMAGE_SIZE) for path in p]
                for image_dir, p in zip(image_dirs, image_paths)
            }

            return d

        def process_depth(path):
            depth_path = os.path.join(path, "depth_images0")
            if os.path.exists(depth_path):
                image_paths = sorted(
                    glob.glob(os.path.join(depth_path, "im_*.png")),
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                return [read_resize_image(path, IMAGE_SIZE) for path in image_paths]
            else:
                return None

        def process_state(path):
            fp = os.path.join(path, "obs_dict.pkl")
            with open(fp, "rb") as f:
                x = pickle.load(f)
            return x["full_state"]

        def process_actions(path):
            fp = os.path.join(path, "policy_out.pkl")
            with open(fp, "rb") as f:
                act_list = pickle.load(f)
            if isinstance(act_list[0], dict):
                act_list = [x["actions"] for x in act_list]
            return act_list

        def process_lang(path):
            fp = os.path.join(path, "lang.txt")
            text = ""  # empty string is a placeholder for missing text
            if os.path.exists(fp):
                with open(fp, "r") as f:
                    text = f.readline().strip()

            return text

        out = dict()

        out["images"] = process_images(episode_path)
        out["depth"] = process_depth(episode_path)
        out["state"] = process_state(episode_path)
        out["actions"] = process_actions(episode_path)
        out["lang"] = process_lang(episode_path)

        # data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
        date_time = datetime.strptime(episode_path.split("/")[-4], "%Y-%m-%d_%H-%M-%S")
        latency_shift = date_time < datetime(2021, 7, 23)

        # shift the actions according to camera latency
        if latency_shift:
            out["images"] = {k: v[1:] for k, v in out["images"].items()}
            out["state"] = out["state"][1:]
            out["actions"] = out["actions"][:-1]
            if out["depth"] is not None:
                out["depth"] = out["depth"][1:]

        # append a null action to the end
        out["actions"].append(np.zeros_like(out["actions"][0]))

        assert len(out["actions"]) == len(out["state"]) == len(out["images"]["images0"])

        # assemble episode
        episode = []
        episode_metadata = dict()

        # map original image name to correct image name according to logged camera topics
        orig_to_new = dict()
        for image_idx in range(len(out["images"])):
            orig_key = orig_names[image_idx]

            if camera_topics[image_idx] in [
                "/cam0/image_raw",
                "/camera0/color/image_raw",
                "/D435/color/image_raw",
            ]:
                # fixed cam should always be image_0
                new_key = "image_0"
                assert new_key[-1] == orig_key[-1], episode_path
            elif camera_topics[image_idx] == "/wrist/image_raw":
                # wrist cam should always be image_3
                new_key = "image_3"
            elif camera_topics[image_idx] in [
                "/cam1/image_raw",
                "/cam2/image_raw",
                "/cam3/image_raw",
                "/cam4/image_raw",
                "/camera1/color/image_raw",
                "/camera3/color/image_raw",
                "/camera2/color/image_raw",
                "/camera4/color/image_raw",
                "/blue/image_raw",
                "/yellow/image_raw",
            ]:
                # other cams can be either image_1 or image_2
                if "image_1" in list(orig_to_new.values()):
                    new_key = "image_2"
                else:
                    new_key = "image_1"
            else:
                raise ValueError(f"Unexpected camera topic {camera_topics[image_idx]}")

            orig_to_new[orig_key] = new_key
            episode_metadata[f"has_{new_key}"] = True

        # record which images are missing
        missing_keys = set(new_names) - set(orig_to_new.values())
        for missing in missing_keys:
            episode_metadata[f"has_{missing}"] = False

        episode_metadata["has_depth_0"] = out["depth"] is not None

        instruction = out["lang"]
        if instruction:
            language_embedding = _embed([instruction])[0].numpy()
        else:
            language_embedding = np.zeros(512, dtype=np.float32)

        for i in range(len(out["actions"])):
            observation = {
                "state": out["state"][i].astype(np.float32),
            }

            for orig_key in out["images"]:
                new_key = orig_to_new[orig_key]
                observation[new_key] = out["images"][orig_key][i]
            for missing in missing_keys:
                observation[missing] = np.zeros_like(out["images"]["images0"][i])
            if episode_metadata["has_depth_0"]:
                observation["depth_0"] = out["depth"][i]
            else:
                observation["depth_0"] = np.zeros(IMAGE_SIZE + (1,), dtype=np.uint8)

            episode.append(
                {
                    "observation": observation,
                    "action": out["actions"][i].astype(np.float32),
                    "discount": 1.0,
                    "reward": float(i == (len(out["actions"]) - 1)),
                    "is_first": i == 0,
                    "is_last": i == (len(out["actions"]) - 1),
                    "is_terminal": i == (len(out["actions"]) - 1),
                    "language_instruction": instruction,
                    "language_embedding": language_embedding,
                }
            )

        episode_metadata["file_path"] = episode_path
        episode_metadata["has_language"] = bool(instruction)

        # create output data sample
        sample = {"steps": episode, "episode_metadata": episode_metadata}

        return episode_path, sample

    for path, camera_topics in paths:
        try:
            yield _parse_examples(path, camera_topics)
        except:
            print(traceback.format_exc())
            yield None


class BridgeDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 10  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = (
        1  # number of paths converted & stored in memory before writing to disk
    )
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = (
        _generate_examples  # handle to parse function from file paths to RLDS episodes
    )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "image_1": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position).",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Side camera RGB observation (varied position)",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "depth_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (1,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera depth observation (fixed position).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot end effector state, consists of [3x XYZ, 3x roll-pitch-yaw, 1x gripper]",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "has_image_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image0 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_1": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image1 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_2": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image2 exists in observation, otherwise dummy value.",
                            ),
                            "has_image_3": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if image3 exists in observation, otherwise dummy value.",
                            ),
                            "has_depth_0": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if depth0 exists in observation, otherwise dummy value.",
                            ),
                            "has_language": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if language exists in observation, otherwise empty string.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""

        # each path is a directory that contains dated directories
        paths = glob.glob(os.path.join(INPUT_PATH, *("*" * (DEPTH - 1))))

        train_filenames, val_filenames = [], []

        for path in paths:
            for dated_folder in os.listdir(path):
                # a mystery left by the greats of the past
                if "lmdb" in dated_folder:
                    continue

                search_path = os.path.join(
                    path, dated_folder, "raw", "traj_group*", "traj*"
                )
                all_traj = glob.glob(search_path)
                if not all_traj:
                    print(f"no trajs found in {search_path}")
                    continue

                config_path = os.path.join(path, dated_folder, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "rb") as f:
                        config = json.load(f)
                    camera_topics = config["agent"]["env"][1]["camera_topics"]
                else:
                    # assumed camera topics if no config.json exists
                    camera_topics = [
                        "/D435/color/image_raw",
                        "/blue/image_raw",
                        "/yellow/image_raw",
                        "/wrist/image_raw",
                    ]
                all_traj = [(t, camera_topics) for t in all_traj]

                random.shuffle(all_traj)
                train_filenames += all_traj[: int(len(all_traj) * TRAIN_PROPORTION)]
                val_filenames += all_traj[int(len(all_traj) * TRAIN_PROPORTION) :]

        print(
            f"Converting {len(train_filenames)} training and {len(val_filenames)} validation files."
        )
        return {
            "train": train_filenames,
            "val": val_filenames,
        }
