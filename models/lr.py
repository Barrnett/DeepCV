#  Copyright © 2020-2020 Hailin Fu All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
"""
Author:
    Hailin Fu, hailinfufu@outlook.com
"""
from deepray.base.layers.core import Linear
from deepray.model.model_ctr import BaseCTRModel


class LogisitcRegression(BaseCTRModel):
    """Instantiates the Deep&Cross Network architecture.

        :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param cross_num: positive integet,cross layer number
        :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
        :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
        :param l2_reg_cross: float. L2 regularizer strength applied to cross net
        :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
        :param init_std: float,to use as the initialize std of embedding vector
        :param seed: integer ,to use as random seed.
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not DNN
        :param dnn_activation: Activation function to use in DNN
        :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
        :return: A Keras model instance.

        """

    def __init__(self, flags):
        super().__init__(flags)

    def build(self, input_shape):
        self.linear_block = self.build_linear()

    def build_network(self, features, is_training=None):
        """

        :param features:
        :param is_training:
        :return:
        """
        linear_logit = self.linear_block(features)
        return linear_logit

    def build_linear(self, hidden=1):
        return Linear(hidden)
