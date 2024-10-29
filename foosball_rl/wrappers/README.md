## Wrapper Configuration
The wrapper configuration file `foosball_rl/wrappers/wrapper_config.yml` contains the following sections:
- `EnvWrapper`: The environment wrappers to use. Each wrapper is applied to the environments separately.
  - `use_add_actions_to_observation_wrapper`: Whether to use the `AddActionToObservationsWrapper` to add the last performed actions to the observation.
  - `use_goal_env_wrapper`: Whether to use the `GoalEnvWrapper` to modify the environment to be compatible with the `GoalEnv` interface for the usage of Hindsight Experience Replay (HerReplayBuffer from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
  - `use_action_space_wrapper`: Whether to use the `DiscreteActionWrapper` or `MultiDiscreteActionWrapper` to discretize the action space.
  - `ActionSpaceWrapper`: Properties for the `DiscreteActionWrapper` or `MultiDiscreteActionWrapper`.
    - `action_space`: The action space to use, either `discrete` or `multi_discrete`. `Continuous` will not use any action space wrapper, as the action space is continuous by nature.
    - `lateral_binning`: The number of bins for the lateral action space.
    - `angular_binning`: The number of bins for the angular action space.
<!-- -->
- `VecEnvWrapper`: The vectorized environment wrappers to use. In contrast to the `EnvWrapper`, the `VecEnvWrapper` is applied to the vectorized environments (i.e. to all parallel environments simultaneously).
  - `use_vec_pbrs_wrapper`: Whether to use the `VecPBRSWrapper` to apply potential-based reward shaping (PBRS) to the environment. An implementation of potential-based reward shaping (PBRS) for the foosball environment. See Ng, A.
  Y., Harada, D., & Russell, S. (1999, June). Policy invariance under reward transformations: Theory and application to reward shaping. In Icml (Vol. 99, pp. 278-287).
  - `use_vec_normalize_wrapper`: Whether to use the `VecNormalize` wrapper to normalize the observations and rewards.
  - `VecNormalizeWrapper`: Properties for the `VecNormalize` wrapper.
    - `norm_obs`: Whether to normalize the observations.
    - `norm_reward`: Whether to normalize the rewards.
    - `clip_obs`: Whether to clip the observations, and if so, the clipping range.
    - `clip_reward`: Whether to clip the rewards, and if so, the clipping range.
  - `use_video_recording_wrapper`: Whether to use the `VideoRecordingWrapper` to record videos of the training process. Works only if the environment's `render_mode=='rgb_array'`, doesn't support human rendering and recording in parallel.
  - `VideoRecordingWrapper`: Properties for the `VideoRecordingWrapper`.
    - `video_length`: The length of the videos in frames.
    - `video_interval`: The frequency of recording videos.
    - `video_log_path_suffix`: The suffix to append to the video log path.
