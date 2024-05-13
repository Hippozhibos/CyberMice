"""CyberMice from Mars"""

import os
import re

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np

_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'assets/CyberMice.xml')

_MICE_MOCAP_JOINTS = []

_UPRIGHT_POS = (0.0, 0.0, 0.0)
_UPRIGHT_QUAT = (1., 0., 0., 0.)
_TORQUE_THRESHOLD = 60

class Mice(legacy_base.Walker):
    """A muscle-controlled mice with control range scaled to [0.1, 1]."""

    def _build(self):
        pass

    @property
    def upright_pose(self):
        pass

    @property
    def mjcf_model(self):
        pass

    @composer.cached_property
    def actuators(self):
        pass

    @composer.cached_property
    def root_body(self):
        """Return the body."""
        return self._mjcf_root.find('body', 'torso')

    @composer.cached_property
    def pelvis_body(self):
        """Return the body."""
        return self._mjcf_root.find('body', 'pelvis')

    @composer.cached_property
    def head(self):
        """Return the head."""
        return self._mjcf_root.find('body', 'skull')

    @composer.cached_property
    def left_arm_root(self):
        """Return the left arm."""
        return self._mjcf_root.find('body', 'scapula_L')

    @composer.cached_property
    def right_arm_root(self):
        """Return the right arm."""
        return self._mjcf_root.find('body', 'scapula_R')

    @composer.cached_property
    def ground_contact_geoms(self):
        """Return ground contact geoms."""
        pass

    @composer.cached_property
    def standing_height(self):
        """Return standing height."""
        pass

    @composer.cached_property
    def end_effectors(self):
        """Return end effectors."""
        pass

    @composer.cached_property
    def observable_joints(self):
        pass

    @composer.cached_property
    def observable_tendons(self):
        pass

    @composer.cached_property
    def mocap_joints(self):
        pass

    @composer.cached_property
    def mocap_joint_order(self):
        pass

    @composer.cached_property
    def bodies(self):
        pass

    @composer.cached_property
    def mocap_tracking_bodies(self):
        pass

    @composer.cached_property
    def primary_joints(self):
        pass

    @composer.cached_property
    def vertebra_joints(self):
        pass

    @composer.cached_property
    def primary_joint_order(self):
        pass

    @composer.cached_property
    def vertebra_joint_order(self):
        pass

    @composer.cached_property
    def egocentric_camera(self):
        pass

    @property
    def _xml_path(self):
        pass

    @composer.cached_property
    def joint_actuators(self):
        pass

    @composer.cached_property
    def joint_actuators_range(self):
        pass

    def pose_to_actuation(self, pose):
        pass

    @composer.cached_property
    def joint_actuator_order(self):
        pass

    def _build_observables(self):
        pass


class RodentObservables(legacy_base.WalkerObservables):
  """Observables for the Rat."""

  @composer.observable
  def head_height(self):
    pass

  @composer.observable
  def sensors_torque(self):
    pass

  @composer.observable
  def tendons_pos(self):
    pass

  @composer.observable
  def tendons_vel(self):
    pass

  @composer.observable
  def actuator_activation(self):
    pass

  @composer.observable
  def appendages_pos(self):
    pass

  @property
  def proprioception(self):
    pass

  @composer.observable
  def egocentric_camera(self):
    pass
    
