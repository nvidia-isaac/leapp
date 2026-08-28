=========
Ecosystem
=========

LEAPP sits between training stacks and deployment stacks. Upstream projects
export a LEAPP bundle from the code they already run. Downstream runtimes
load that bundle instead of rebuilding observation, action, and state
handling by hand.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Isaac Lab
      :link: https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/policy_deployment/05_leapp/exporting_policies_with_leapp.html
      :link-type: url
      :img-bottom: _static/images/ecosystem_isaac_lab.gif
      :img-alt: Isaac Lab
      :class-card: sd-rounded-3
      :class-body: leapp-card-sm

      Export a trained reinforcement learning policy from Isaac Lab as a
      LEAPP bundle. The export keeps the observation preprocessing, action
      postprocessing, and recurrent state used during training.

   .. grid-item-card:: Isaac ROS Deploy
      :link: https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html
      :link-type: url
      :img-bottom: _static/images/ecosystem_isaac_ros.gif
      :img-alt: Isaac ROS Deploy
      :class-card: sd-rounded-3
      :class-body: leapp-card-sm

      Run a LEAPP bundle on ROS 2. The runtime loads the bundle, runs
      inference, maps policy terms onto topics or ``ros2_control``, and can
      gate outputs through a safety controller.
