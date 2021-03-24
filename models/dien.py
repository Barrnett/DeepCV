
"""
Author:
    Hailin Fu, hailinfufu@outlook.com
"""

from absl import flags

from deepray.model.model_ctr import BaseCTRModel

FLAGS = flags.FLAGS


class DeepInterestEvolutionNetwork(BaseCTRModel):

    def __init__(self, flags):
        super(DeepInterestEvolutionNetwork, self).__init__(flags)

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """
