# Environment Configuration
#### The environment configuration for each implemented scenario is described in detail in the README.md file in the respective environment folder.
Each environment configuration contains the following sections:
 - `Environment`: Defines the environment parameters:
    - `horizon`: The maximum number of steps per episode.
    - `step_frequency`: The action is repeated for `step_frequency` simulation steps before a new action is taken, e.g. with the goalkeeper environment, the simulation step frequency is determined by `timestep="0.002"`, which means that 500 simulation steps are executed per second. Setting `step_frequency=10` would mean that the action is repeated for 10 simulation steps before a new action is taken.
    - `render_mode`: The render mode of the environment. The render mode can be set to `null`, `rgb_array` or `human`. `null` will not render the environment at all, `rgb_array` will render the environment as an RGB array (required if image observations are desired) and `human` will render a live interactive view of the training, where the camera can be moved freely and the speed of the simulation can be adjusted. Only one environment is supported with `human` rendering (i.e. no parallel training and no rendering of the evaluation environment). 
    - `use_image_obs`: Whether to use image observations (requires `render_mode=rgb_array`).
<!-- -->
 - `EpisodeDefinition`: Defines the episode definition:
    - `..<<params>>..`: Here, several parameters can be defined that can be used for the episode definition. The already implemented episode definitions will be discussed in the respective environment readme-files. Decoupling the episode definition from the environment allows more flexibility in defining various episode definitions for a single simulated environment.
