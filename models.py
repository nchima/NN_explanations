from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout, Activation , Conv2D, MaxPooling2D
from keras.models import Model


def vgg_16_imagenet():
 
  x = Input((3, 224, 224))

  f = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
  f = Conv2D(64, (3, 3), padding='same', activation='relu')(f)
  f = MaxPooling2D((2,2))(f)

  f = Conv2D(128, (3, 3), padding='same', activation='relu')(f)
  f = Conv2D(128, (3, 3), padding='same', activation='relu')(f)
  f = MaxPooling2D((2,2))(f)

  f = Conv2D(256, (3, 3), padding='same', activation='relu')(f)
  f = Conv2D(256, (3, 3), padding='same', activation='relu')(f)
  f = Conv2D(256, (3, 3), padding='same', activation='relu')(f)
  f = MaxPooling2D((2,2))(f)

  f = Conv2D(512, (3, 3), padding='same', activation='relu')(f)
  f = Conv2D(512, (3, 3), padding='same', activation='relu')(f)
  f = Conv2D(512, (3, 3), padding='same', activation='relu')(f)
  f = MaxPooling2D((2,2))(f)

  f = Conv2D(512, (3, 3), padding='same', activation='relu')(f)
  f = Conv2D(512, (3, 3), padding='same', activation='relu')(f)
  f = Conv2D(512, (3, 3), padding='same', activation='relu')(f)
  f = MaxPooling2D((2,2))(f)

  f = Flatten()(f)
  f = Dense(4096, activation='relu')(f)
  f = Dense(4096, activation='relu')(f)

  f = Dense(1000)(f)

  model = Model(x, f)

  try:
    model.load_weights('applications_vgg16_weights.h5')
  except:
     weight_model = VGG16()
     weight_model.save_weights('applications_vgg16_weights.h5')
     model.load_weights('applications_vgg16_weights.h5')

  return model
