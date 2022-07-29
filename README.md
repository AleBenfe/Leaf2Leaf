# Leaf2Leaf

Python code for Leaf2Leaf architecture described in 

A. Benfenati, D. Bolzi, P.Causin and R. Oberti, *A Deep Learning Generative Model Approach for Image Synthesis of Plant Leaves*

The code is based on Tensorflow and Keras.

The repositoty has 3 directories:

- [ResVae](https://github.com/AleBenfe/Leaf2Leaf#resvae)
- pix2pix
- Leaf2LeafTranslation

## ResVae

The Residual Variational AutoEncoder for the leaf skeleton generation is in `RES_VAE.ipynb`. The trained model described in the paper is saved in the `Results` directory. The code is tailored for being used also on Colab.

The result of the pretrained model is downbelow: on the left the training sample is shown, whilst on the right the recovered image by the ResVAE is depicted.

![Image](https://github.com/AleBenfe/Leaf2Leaf/blob/main/Figures/ex_ResVae.png "Results of ResVAE")

### Creation of a new syinthetic sample

The following commands load the pretrained decoder, which is used as the generator of a leaf vein patter (the same code can be found in `RES_VAE.ipynb`).
```
json_file           = open('Results/decoder.json', 'r')
loaded_decoder_json = json_file.read()
loaded_decoder      = model_from_json(loaded_decoder_json)
loaded_decoder.load_weights("Results/decoder.h5")
json_file.close()

to_build  = tf.random.normal(shape = [1, latent_dim])
generated = loaded_decoder.predict(to_build)
```

One of the possible results is depicted below
![Image](https://github.com/AleBenfe/Leaf2Leaf/blob/main/Figures/ex_generated.png "Example of generated image")

## pix2pix

A conditional Generative Adversarial Network (cGAN) is employed for the development of a generator of a complete leaf starting from a vein pattern. The subdirectories are

* `train/A`: it contains the training samples of vein patterns
* `train/B`: it contains the training samples of leaf RGB images
* `train/C`: it contains the training samples of leaf RGNIR images. For employing RGNIR images, change 

```
path2 = path + '/B'
```

into 

```
path2 = path + '/C'
```
 in the second cell of `pix2pix.ipynb` 
* `Results`: it contains the pretrained model

Down below one possible result of the generator: on the left, a vein pattern that was not employed for the training, on the right the generated complete leaf

![Image](https://github.com/AleBenfe/Leaf2Leaf/blob/main/Figures/not_seen.png "Unseen veins pattern") ![Image](https://github.com/AleBenfe/Leaf2Leaf/blob/main/Figures/generated.png "Complete generated image")

## Leaf2LeafTranslation

End-to-End procedure: a synthetic leaf veins pattern is created, which is then passed to the generator of the cGAN.  
* `End-to-End.ipynb`: code of the entire procedure, it loads the model `decoder.json` and the weights `decoder.h5` of ResVAE's decoder and the generator of the cGAN `model012000.h`. 
* `model_nir_012000.h5`: it contains the cGAN's generator for RGNIR images.

Down below a possible result is shown: on the left, a synthetic veins pattern created by the decoder of the ResVAE, on the right the complete generated leaf.

 ![Image](https://github.com/AleBenfe/Leaf2Leaf/blob/main/Figures/e2e.png "Unseen veins pattern") 



 
