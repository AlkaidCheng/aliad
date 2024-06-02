from typing import Optional, Union, List, Tuple

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (BatchNormalization, Activation,
                                     Conv3D, Dense, Dropout)

from quickstats import semistaticmethod, AbstractObject
from quickstats.utils.common_utils import combine_dict

from aliad.interface.tensorflow.layers import EdgeConv

class MultiParticleNet(AbstractObject):

    DEFAULT_CONFIG = {
        "num_points"  : 300,
        "num_jets"    : 2,
        "num_class"   : 1,
        "K"           : 16,
        "conv_params" : [(64, 64, 64),
                         (128, 128, 128),
                         (256, 256, 256)],
        "fc_params"   : [(256, 0.3)],
        "pool_method" : "average",
        "batchnorm"   : True,
        "activation"  : "relu"
    }
    
    def __init__(self, config:Optional[dict]=None,
                 verbosity:str="INFO"):
        super().__init__(verbosity=verbosity)
        self.config = combine_dict(self.DEFAULT_CONFIG, config)
    
    def get_model(self, points, features=None, masks=None,
                  jet_features=None, param_features=None,
                  param_scale:float=1.0, name='ParticleNet'):
        # points       : (nevent, njet, nparticle, ncoords)
        # features     : (nevent, njet, nparticle, nfeatures)
        # masks        : (nevent, njet, nparticle, 1)
        # jet_features : (nevent, njet, njetfeatures)
    
        with tf.name_scope(name):
            if features is None:
                features = points
    
            if masks is not None:
                coord_shift = tf.multiply(99., tf.cast(tf.expand_dims(masks, axis=-1), dtype='float32'))
            else:
                coord_shift = None
            fts = tf.expand_dims(features, axis=-2)
            fts = BatchNormalization(name=f"{name}_fts_bn")(fts)
            fts = tf.squeeze(fts, axis=-2)
            for layer_idx, layer_channels in enumerate(self.config['conv_params']):
                pts = points if layer_idx == 0 else fts
                if masks is not None:
                    pts = tf.add(coord_shift, pts)
                fts = EdgeConv(pts, fts, channels=layer_channels,
                               K=self.config['K'], 
                               batchnorm=self.config['batchnorm'],
                               activation=self.config['activation'],
                               pooling=self.config['pool_method'],
                               name=f'{name}_EdgeConv{layer_idx}')
    
            if masks is not None:
                fts_mask = tf.cast(tf.math.logical_not(tf.expand_dims(masks, axis=-1)), dtype='float32')
                fts = tf.multiply(fts, fts_mask)

            # shape = (nevent, njet, nchannel)
            pool = tf.reduce_mean(fts, axis=-2)
            
            components = [pool]
            if jet_features is not None:
                components.append(jet_features)
            if param_features is not None:
                param_features_ = tf.expand_dims(param_features, axis=-1)
                param_features_ = tf.multiply(param_features_, param_scale)
                components.append(param_features_)
            out = tf.concat(components, -1)
            out = tf.reshape(out, (-1, tf.reduce_prod(tf.shape(out)[1:])))

            #if self.config['batchnorm']:
            #    out = BatchNormalization(name='bn_concat')(out)
                
            if self.config['fc_params'] is not None:
                x = out
                for layer_idx, layer_param in enumerate(self.config['fc_params']):
                    units, drop_rate = layer_param
                    x = Dense(units, activation='relu')(x)
                    if drop_rate is not None and drop_rate > 0:
                        x = Dropout(drop_rate)(x)
                out = Dense(self.config['num_class'], activation='sigmoid')(x)
                
            inputs = [points, features]
            if masks is not None:
                inputs.append(masks)
            if jet_features is not None:
                inputs.append(jet_features)
            if param_features is not None:
                inputs.append(param_features)
        model = Model(inputs=inputs, outputs=out, name=name)
        return model