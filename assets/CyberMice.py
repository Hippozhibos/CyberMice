"""CyberMice from Mars"""

import os
import re

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.composer import variation
from dm_control.composer import Arena
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import bowl

from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks
from dm_control import suite
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np

import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image

# Use svg backend for figure rendering
# %config InlineBackend.figure_format = 'svg'

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Inline video helper function
if os.environ.get('COLAB_NOTEBOOK_TEST', False):
  # We skip video generation during tests, as it is quite expensive.
  display_video = lambda *args, **kwargs: None
else:
  def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())

# Seed numpy's global RNG so that cell outputs are deterministic. We also try to
# use RandomState instances that are local to a single cell wherever possible.
np.random.seed(42)


_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'CyberMice.xml')
class Mice(composer.Entity):
    """A muscle-controlled mice with control range scaled to [0.1, 1]."""

    def _build(self):
        self._mjcf_model = mjcf.from_path(_XML_PATH)     

    @property
    def mjcf_model(self):
        """Return the model root."""
        return self._mjcf_model


    @property
    def actuators(self):
        """Return all actuators."""
        return tuple(self._mjcf_model.find_all('actuator'))

    def _build_observables(self):
        return MiceObservables(self)

NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.

class MiceObservables(composer.Observables):
  """Observables for the Mice."""
  @composer.observable
  def joint_positions(self):
     all_joints = self._entity.mjcf_model.find_all('actuator')
     return observable.MJCFFeature('ctrl', all_joints)
  
class PressWithSpecificForce(composer.Task):
  def __init__(self, creature):
    self._creature = creature
    self._arena = floors.Floor()
    self._arena.add_free_entity(self._creature)
    self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

    # Configure initial poses
    self._creature_initial_pose = (0, 0, 0.15)

    # Configure variators
    self._mjcf_variator = variation.MJCFVariator()
    self._physics_variator = variation.PhysicsVariator()

    # Configure and enable observables
    self._creature.observables.joint_positions.enabled = True

    self.control_timestep = NUM_SUBSTEPS * self.physics_timestep
    self._task_observables = {}

    # Adjust offscreen framebuffer width
    self._adjust_offscreen_framebuffer_width(1600)  # Set to desired width

    # Adjust camera settings
    self._adjust_camera_distance(0.3)  # Set to desired distance

  @property
  def root_entity(self):
    return self._arena

  @property
  def task_observables(self):
    return self._task_observables

  def initialize_episode_mjcf(self, random_state):
    self._mjcf_variator.apply_variations(random_state)

  def initialize_episode(self, physics, random_state):
    self._physics_variator.apply_variations(physics, random_state)

  def get_reward(self, physics):
    return 0.0
  
  def _adjust_offscreen_framebuffer_width(self, width):
    body_element = self._arena.mjcf_model.find('body', 'root')
    if body_element is not None:
        worldbody_element = body_element.find('worldbody')
        if worldbody_element is None:
            worldbody_element = mjcf.element.Element('worldbody')
            body_element.add(worldbody_element)
        visual_element = worldbody_element.find('visual')
        if visual_element is None:
            visual_element = mjcf.element.Element('visual')
            worldbody_element.add(visual_element)
        global_element = visual_element.find('global')
        if global_element is None:
            global_element = mjcf.element.Element('global')
            visual_element.add(global_element)
        global_element.set_attrib('offwidth', str(width))

  def _adjust_camera_distance(self, distance):
    camera_element = self._arena.mjcf_model.find('camera', 'camera')
    if camera_element is not None:
        camera_element.set_attrib('distance', str(distance))

         # Find the position of your model (assuming it's named 'your_model_name')
        model_position = self._creature.mjcf_model.find('body', 'CyberMice').pos
        
        # Set camera lookat to the position of your model
        camera_element.set_attrib('lookat', ' '.join(map(str, model_position)))




creature = Mice()
task = PressWithSpecificForce(creature)
env = composer.Environment(task, random_state=np.random.RandomState(42))

env.reset()

# PIL.Image.fromarray(env.physics.render())
image = PIL.Image.fromarray(env.physics.render())

image.save("output_image.png")