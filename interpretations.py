from explanations import AumannShapley, Explainer, Explanation

from keras.models import Model
from scipy.ndimage.filters import gaussian_filter
from theano import grad, function

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


class Interpreter(object):
  '''
  Interprets explanations via the input influence on the relevant neurons.

  YOU DO NOT NEED TO MODIFY THIS CODE.
  '''
  def __init__(self):
    self.prev_influence_measure_params = None
    self.influence_measure = AumannShapley(ignore_difference_term=False)

  def interpret_input(self, 
      explanation, 
      output_file=None, 
      pos_from_top=0,
      blur=4, 
      thresh=0.6):
    '''
    Interprets the explanation as an image. If output_file is given, the image
    is saved in the output_file. Otherwise the image is displayed inline.

    :blur | amount of blurring to apply to the mask. Increasing the blur will
      tend to make the visualization smoother.
    :thresh | the percentile cutoff for the mask. The closer to 1.0 this is, the
      more focused the explanation will be.
    '''

    # This is already an input-explanation, so we can just visualize the
    # explanation directly.
    input_inf = explanation.influences.mean(axis=1)

    img = self.__process_image(explanation.instances, input_inf, blur, thresh)

    plt.figure(figsize=(img.shape[0] / 16 , img.shape[1] / 16))
    plt.imshow(img)
    if output_file:
      plt.savefig(output_file, bbox_inches='tight')

  def interpret_neuron(self, 
      explanation,
      neuron_index,
      output_file=None, 
      blur=4, 
      thresh=0.6):
    '''
    Interprets the explanation as an image. If output_file is given, the image
    is saved in the output_file. Otherwise the image is displayed inline.

    To visualize the explanation, the input influence on each of the relevant
    neurons is calculated. The input influence is then blurred to make the
    visualization smoother, and thresholded to create a mask that shows only the
    most relevant parts of the image according to the influence.

    :neuron_index | the index of the neuron to be visualized. Note that this is
      a flattened index, so if the layer being explained contains feature maps
      rather than a vector of fully-connected neurons, the index refers to the
      index of the neuron when the feature map is flattened.
    :blur | amount of blurring to apply to the mask. Increasing the blur will
      tend to make the visualization smoother.
    :thresh | the percentile cutoff for the mask. The closer to 1.0 this is, the
      more focused the explanation will be.
    '''

    # 'Cache' the influence measure so we don't have to compile if it's not
    # necessary. This should make playing around with the blur and thresh
    # parameters less expensive.
    prev_influence_measure_params = (
      'neuron', explanation._model, explanation._layer, neuron_index)
    if self.prev_influence_measure_params != prev_influence_measure_params:
      self.prev_influence_measure_params = prev_influence_measure_params

      Q = (explanation._model.layers[explanation._layer - 1]
        .output[:, neuron_index]
        .sum())
      x = explanation._model.input

      self.influence_measure.compile(Q, x)

    input_inf = self.influence_measure(explanation.instances).mean(axis=1)

    img = self.__process_image(explanation.instances, input_inf, blur, thresh)

    plt.figure(figsize=(img.shape[0] / 16 , img.shape[1] / 16))
    plt.imshow(img)
    if output_file:
      plt.savefig(output_file, bbox_inches='tight')

  def interpret_feature_map(self, 
      explanation, 
      feature_map_index,
      output_file=None, 
      blur=4, 
      thresh=0.6):
    '''
    Interprets the explanation as an image. If output_file is given, the image
    is saved in the output_file. Otherwise the image is displayed inline.

    To visualize the explanation, the input influence on each of the relevant
    neurons is calculated. The input influence is then blurred to make the
    visualization smoother, and thresholded to create a mask that shows only the
    most relevant parts of the image according to the influence.

    :feature_map_index | the index of the feature map to be visualized. The
      neuron with maximum influence in the given feature map will be visualized.
    :blur | amount of blurring to apply to the mask. Increasing the blur will
      tend to make the visualization smoother.
    :thresh | the percentile cutoff for the mask. The closer to 1.0 this is, the
      more focused the explanation will be.
    '''

    # 'Cache' the influence measure so we don't have to compile if it's not
    # necessary. This should make playing around with the blur and thresh
    # parameters less expensive.
    prev_influence_measure_params = (
      'map', explanation._model, explanation._layer, feature_map_index)
    if self.prev_influence_measure_params != prev_influence_measure_params:
      self.prev_influence_measure_params = prev_influence_measure_params

      Q = (explanation._model.layers[explanation._layer - 1]
        .output[:, feature_map_index]
        .max(axis=-1)
        .max(axis=-1)
        .sum())
      x = explanation._model.input

      self.influence_measure.compile(Q, x)

    input_inf = self.influence_measure(explanation.instances)
    input_inf = input_inf.mean(axis=1)

    img = self.__process_image(explanation.instances, input_inf, blur, thresh)

    plt.figure(figsize=(img.shape[0] / 16 , img.shape[1] / 16))
    plt.imshow(img)
    if output_file:
      plt.savefig(output_file, bbox_inches='tight')

  def __process_image(self, original, influences, blur, thresh):
    
    # Blur the influences so the explanation is smoother.
    influences = [gaussian_filter(influence, blur) for influence in influences]
    
    # Normalize the influences to be in the range [0, 1].
    influences = [influence - influence.min() for influence in influences]
    influences = [
      0. * influence if influence.max() == 0. else influence / influence.max()
      for influence in influences]

    # Threshold them to create a mask.
    masks = [influence > thresh for influence in influences]

    # Do some final blurring to make the visualization easy to look at.
    masks = [
      np.minimum(
        (gaussian_filter(mask.astype('float32'), blur / 2.) > 0.01)
          .astype('float32') + 0.2, 
        1) for mask in masks]

    # Transpose the image and the influences so they match plt's dimension
    # ordering.
    original = original.transpose(0,2,3,1) / 255
    masks = np.expand_dims(masks, 1).transpose(0,2,3,1)
    
    # Use the mask on the original image to visualize the explanation.
    return np.concatenate(original) * np.concatenate(masks)
