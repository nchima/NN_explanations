from keras.models import Model
from scipy.ndimage.filters import gaussian_filter
from theano import grad, function

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np


## Influence ###################################################################

class AumannShapley():

  def __init__(self, resolution=50, ignore_difference_term=False):
    '''
    The parameters here allow us to generalize to many different methods. E.g.,

      saliency maps:         resolution=1,  ignore_difference_term=True
      deep taylor:           resolution=2,  ignore_difference_term=False
      integrated gradients:  resolution=50, ignore_difference_term=False
    '''
    self.res = resolution
    self.ignore_difference_term = ignore_difference_term

  def compile(self, Q, z):
    '''
    Compiles a theano function for the gradient of the quantity of interest, Q, 
    with respect to the input tensor, z.

    Q must be a scalar tensor.

    YOU DO NOT NEED TO MODIFY THIS CODE.
    '''
    self.dF = function([z], grad(Q, wrt=z), allow_input_downcast=True)

    return self

  def __call__(self, instances, baseline=None):
    '''
    Returns the approximate Aumann Shapley value (integrated gradients) for the
    given batch of instances, with respect to the given baseline.

    If baseline is None, then it is assumed to be 0.

    The resolution for approximate AS is given in self.res.

    If self.ignore_difference_term is true, the integrated gradients are not
    multiplied by the difference between the instances in the baseline.
    Otherwise, the result is calculated as in the integrated gradients paper.

    You may make use of self.dF, which is a function that, given a batch of
    instances, returns a batch of the gradients of the compiled quantity of
    interest with respect to the activations, z, at the desired layer; evaluated
    at each instance in the batch.

    Returns a numpy array with the influences for each instance in the batch.

    TODO: implement this.
    ''' 

    if (baseline is None):
      baseline = np.zeros(instances.shape)

    # grads = self.dF(instances, z)
    # constant = np.mean(grads[0:self.res])
    # diff = lambda x1, x2: x1 - x2
    # vecdiff = np.vectorize(diff)
    # differences = vecdiff(instances,z)
    # return (differences * constant)

    m = self.res
    # m = 5

    deriv = []

    for k in range (m):

      var = (float(k+1)/float(m)) * (instances-baseline) + baseline
      deriv.append(self.dF(var))
      
    deriv = np.asarray(deriv)
    # print("deriv.shape ", deriv.shape)    
    IG = np.mean(deriv, axis=0)

    if (self.ignore_difference_term == True):
      # print("IG.shape ", IG.shape)
      return IG
    else:
      return(instances-baseline)*IG

    # raise NotImplementedError('You need to implement this function.')


## Explanation #################################################################

class Explainer(object):
  '''
  Class for explaining keras models using influence-directed explanations.
  '''

  def __init__(self, keras_model, layer, influence_measure):
    '''
    Begins creation of an Explainer that can produce explanations for the given
    keras_model, sliced at the given layer, using the given attribution method.

    YOU DO NOT NEED TO MODIFY THIS CODE.
    '''

    # Checks on layer for debugging.
    if layer >= len(keras_model.layers):
      raise ValueError(
        'The layer given must be in range for the given model. '
        'You passed layer {}, but there were only {} layers in the model.'
        .format(layer, len(keras_model.layers)))

    # Save the arguments.
    self.model = keras_model
    self.layer = layer
    self.influence_measure = influence_measure

    # Make a preprocessor to compute the input activations to the target layer.
    self.preprocess = (
      (lambda X : X) if layer == 0 else
      Model(
        keras_model.input, 
        keras_model.layers[layer - 1].output)
      .predict)

  def _get_absolute_quantity_of_interest(self, c):
    '''
    Returns a symbolic tensor representing the pre-softmax output of the model
    for class c. This is the absolute quantity of interest for class c.

    TODO: implement this.
    '''

    # input placeholder
    outputs = self.model.layers[-1].output
    
    return outputs[:,c].sum() 
    # raise NotImplementedError('You need to implement this function.')

  def _get_comparative_quantity_of_interest(self, c1, c2):
    '''
    Returns a symbolic tensor representing the pre-softmax output of the model
    for class c1 minus the pre-softmax output of the model for class c2. This is
    the comparative quantity of interest for class c1 vs. class c2.

    TODO: implement this.
    '''

    outputs = self.model.layers[-1].output
    
    return outputs[:,c1].sum() - outputs[:,c2].sum() 


    # raise NotImplementedError('You need to implement this function.')

  def for_absolute_qoi(self, c):
    '''
    Returns an explainer for the absolute quantity of interest for class c.

    YOU DO NOT NEED TO MODIFY THIS CODE.
    '''
    return _ExplainerForQuantity(
      self._get_absolute_quantity_of_interest(c),
      self.preprocess,
      self.model,
      self.layer,
      self.influence_measure)

  def for_comparative_qoi(self, c1, c2):
    '''
    Returns and explainer for the comparative quantity of interest for class c1
    vs. c2.
    
    YOU DO NOT NEED TO MODIFY THIS CODE.
    '''
    return _ExplainerForQuantity(
      self._get_comparative_quantity_of_interest(c1, c2),
      self.preprocess,
      self.model,
      self.layer,
      self.influence_measure)


class _ExplainerForQuantity(object):
  '''
  YOU DO NOT NEED TO MODIFY THIS CODE. Furthermore, you should not instantiate
  this class directly."
  '''
  def __init__(self, Q, h, model, layer, influence_measure):
    self.h = h
    self.model = model
    self.layer = layer

    z = model.layers[layer].input

    self.influence_measure = influence_measure.compile(Q, z)

  def explain(self, X, baseline=None):
    Z = self.h(X)
    
    if baseline is None:
      baseline = np.zeros(Z.shape)

    return Explanation(
      self.influence_measure(Z, baseline), 
      X,
      self.model, 
      self.layer)

  def explain_instance(self, x, baseline=None):
    return self.explain(np.expand_dims(x, axis=0), baseline)


class Explanation(object):
  '''
  YOU DO NOT NEED TO MODIFY THIS CODE. Furthermore, you should not instantiate
  this class directly."
  '''
  def __init__(self, influences, instances, model, layer):
    self.influences = influences
    self.instances = instances
    self._model = model
    self._layer = layer


## Interpretation ##############################################################

class VisualInfluenceInterpreter(object):
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
      blur=3, 
      thresh=0.75):
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
      blur=3, 
      thresh=0.75):
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

      Q = (
        explanation._model.layers[explanation._layer - 1]
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
      blur=3, 
      thresh=0.75):
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

      Q = (
        explanation._model.layers[explanation._layer - 1]
        .output[:, feature_map_index]
        .max())
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

    # Transpose the image and the influences so they match plt's dimension
    # ordering.
    original = original.transpose(0,2,3,1)
    masks = np.expand_dims(masks, 1).transpose(0,2,3,1)
    
    # Use the mask on the original image to visualize the explanation.
    return np.concatenate(original) * np.concatenate(masks)
