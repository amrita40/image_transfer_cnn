# Image Style Transfer Using CNNs

A Flask application that implements neural style transfer, allowing users to blend content and style images seamlessly. The application leverages the power of PyTorch for the style transfer algorithm and uses Tailwind CSS for modern, responsive styling. This combination ensures both high-performance image processing and an intuitive, visually appealing user interface.

Additionally, a camera module is added whereby users can submit content images through their webcam.

[Medium Blog About the project](https://medium.com/@sureshnithin1729/image-style-transfer-using-cnns-8ba3041df359)

## Demo Video


https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/9ae61e41-accf-43e1-b851-4b71ba304cb3



[Watch the Demo Video on YouTube](https://www.youtube.com/watch?v=6c5A9ZEjpB8)

## Screenshots

![Screenshot from 2024-06-14 23-37-51](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/2fd68a5e-973b-4f99-9593-c85fdaf080e4)
![Screenshot from 2024-06-14 23-38-12](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/d8ca06bc-e472-4af5-9752-fbc34e9dc544)
![Screenshot from 2024-06-14 23-38-21](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/791eda31-668a-47e9-84d6-4a8b7288e065)
![Screenshot from 2024-06-14 23-38-41](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/6bffac62-4ce9-484f-86e8-fe6c409242e9)

![Screenshot from 2024-06-14 23-38-48](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/e6212818-b730-44c3-b380-06fffdaf6d77)
![Screenshot from 2024-06-14 23-39-50](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/6a2f1323-42f7-48f0-81b4-97ad8d67216a)
![Screenshot from 2024-06-14 23-40-16](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/b944b8b7-9b0c-49d6-982a-f82552d462b2)
![Screenshot from 2024-06-14 23-40-26](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/a40a260b-8794-441a-8b22-24509eccdd8f)
![Screenshot from 2024-06-14 23-40-33](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/a14ff611-1b71-4323-9a99-23064c6a148e)


## What is Image Style Transfer?
Image style transfer, also known as Neural Style Transfer, refers to a category of software algorithms that modify digital images or videos to emulate the visual style of another image. Essentially, this technique involves combining two images—a content image and a style reference image (such as a famous artwork)—to produce a new image that maintains the content of the first image but adopts the visual style of the second.

Examples : 
![NYC-scenederue](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/c8180b75-25e6-4954-bd27-82e19a155aab)
![victoria-memorial-womanwithhat-matisse](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/85eb0651-dafa-47c9-8f2a-1ba20d5f4365)
![boat-womaninpeasantdress](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/dbbdd6f1-56fd-49d6-8d3b-01a938121cee)

![girl-brushstrokes](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/de4eac43-550a-431f-9c3f-3724d4040447)

![lenna-picasso-seatednude](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/fcf20ac5-97ab-4421-8f44-304d1ff60a2b)

![montreal](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/cc9ec4ee-1553-45aa-b815-0d2af3c0bb80)

## Research Paper 
In this project, we will replicate the style transfer method described in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch.

## VGG19 Architecture
For this style transfer project, we utilize the features extracted from the 19-layer VGG Network, which includes a sequence of convolutional and pooling layers, along with several fully-connected layers. The convolutional layers are labeled by their stack and sequence within the stack, such as `Conv_1_1` for the first layer in the first stack, and `Conv_2_1` for the first layer in the second stack. The deepest convolutional layer is `Conv_5_4`.

![Screenshot from 2024-06-10 12-51-05](https://github.com/Nithin1729S/Image-Style-Transfer-Using-CNNs/assets/78496667/06904212-117f-468c-9de0-a302c9b001cc)


## Separating Style and Content
The process of style transfer involves distinguishing between the content and style of an image. Given a content image and a style image, the goal is to generate a new image that combines the elements of both:
- The arrangement and objects are similar to the content image.
- The style, including colors and textures, resembles the style image.

In this project, we will use a pre-trained VGG19 network to extract the content and style features from an image.



## Load Features in VGG19
The VGG19 network is divided into two parts:
- `vgg19.features`, which contains all convolutional and pooling layers.
- `vgg19.classifier`, which includes the three linear classifier layers at the end.

We will only need the features part, and we will "freeze" the weights of these layers.

## Load in Content and Style Images
You can use any images you like.It's beneficial to use smaller images and adjust the content and style images to be the same size.

## VGG19 Layers
To obtain the content and style representations of an image, we pass the image through the VGG19 network to reach the desired layers and then extract the output from those layers.

## Gram Matrix
The output of each convolutional layer is a Tensor with dimensions related to the batch size, depth (d), height, and width (h, w). The Gram matrix of a convolutional layer can be computed as follows:
1. Obtain the tensor's depth, height, and width using `batch_size, d, h, w = tensor.size()`.
2. Reshape the tensor so that the spatial dimensions are flattened.
3. Compute the Gram matrix by multiplying the reshaped tensor by its transpose.

## Putting it all Together
With functions for feature extraction and Gram matrix computation in place, we can now integrate these components. We'll extract features from our images and calculate the Gram matrices for each layer in our style representation..

## Loss and Weights
### Individual Layer Style Weights
Below, you have the option to weight the style representation at each relevant layer. Using a range between 0–1 for these weights is suggested. Emphasizing earlier layers (conv1_1 and conv2_1) will result in more prominent style artifacts in the final image. Emphasizing later layers will highlight smaller features because each layer varies in size, creating a multi-scale style representation

### Content and Style Weight
Following the method in the paper, we define alpha (content_weight) and beta (style_weight). This ratio influences the level of stylization in the final image. It is recommended to keep content_weight = 1 and adjust the style_weight to achieve the desired effect.

```python
# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {
    'conv1_1': 1.,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1
}

content_weight = 1 # alpha
style_weight = 1e6 # beta
```

## Updating the Target & Calculating Losses
You will decide on the number of steps for updating your image, similar to a training loop. The number of steps is flexible, but at least 2000 steps are recommended for good results. Fewer steps might be sufficient for testing different weight values or experimenting with images.

During each iteration, calculate the content and style losses and update your target image accordingly.

## Content Loss
The content loss is the mean squared difference between the target and content features at layer conv4_2, calculated as follows:

```python 
content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
```
## Style Loss
The style loss is calculated similarly but involves iterating through multiple layers specified by the style_weights dictionary. Compute the Gram matrix for the target image (target_gram) and the style image (style_gram) at each layer and compare them to calculate the layer_style_loss. This value is then normalized by the size of the layer.

## Total Loss
Finally, create the total loss by combining the style and content losses, weighted by the specified alpha and beta values. Print this loss periodically; even if it starts large, it should decrease over iterations. Focus on the appearance of the target image rather than the loss value itself.

## Kaggle Notebook
You can view the basic implementation of the project in this kaggle notebook. https://www.kaggle.com/code/nithin1729s/styletransfer
## Running the Project

1. Set up a virtual environment:

    ```bash
    virtualenv env
    source env/bin/activate
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Install tkinter
   ```bash
   sudo apt-get install python3-tk
   ```
4. Run the project:

    ```bash
    python wsgi.py
    ```



