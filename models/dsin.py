

"""
Author:
    Hailin Fu, hailinfufu@outlook.com
"""
from deepray.model.model_ctr import BaseCTRModel


class DeepSessionInterestNetwork(BaseCTRModel):
    def __init__(self):
        super(DeepSessionInterestNetwork, self).__init__()

    def build_network(self, features, is_training=None):
        """
        TODO

        :param features:
        :param is_training:
        :return:
        """
