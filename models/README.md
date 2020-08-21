### Generator

We adopt the 'U-Net' as our G, which consists of 1 input layer, 7 encoding layers, 1 bottleneck layer, 7 decoding layers, and 1 output layer. For each encoding-decoding layer pair, we add a direct skip from the encoding layer to the decoding layer. Details about filter size, activation function, drop out setting and so on can be accessed through the model.py file.

![](https://github.com/caoxingdong/adTMO/blob/master/models/imgs/G.png?raw=true)

### Discriminator

![](https://github.com/caoxingdong/adTMO/blob/master/models/imgs/D.png?raw=true)

We adopt the 70 * 70 PatchGAN as our D. Each element in the output matrix maps to a 70 * 70 receptive field in the input
layer, identifying this patch as real or fake. Details about filter size, activation function, drop out setting and so on can be accessed through the model.py file.



### Loss function

We use adversarial loss, feature matching loss and perceptual loss to train our **adTMO**.
$$
\mathcal L_{cGAN}(G,D) = \mathbb{E}_{(x,y)} \log D(x,y) + \mathbb{E}_{(x)} \log(1-D(x,G(x)))\\
\mathcal L_{FM}(G,D) = \mathbb{E}_{(x,y)} \sum_{i=1}^{M} \frac{1}{U_i}[||D^{(i)}(x,y) - D^{(i)}(x,G(x))||_1]\\
\mathcal L_{prp}(G) = \mathbb{E}_{(x,y)} \sum_{i=1}^{N} \frac{1}{V_i} [||F^{(i)}(y) - F^{(i)}(G(x))||_1]
$$
The overall objective can be expressed as
$$
G^* = \text{arg} \mathop{\text{min}}\limits_{G} \mathop{\text{max}}\limits_{D}  \mathcal L_{cGAN}(G,D)  +  \alpha \mathcal L_{FM}(G,D)  +  \beta \mathcal L_{prep}(G)
$$
More details can be accessed through the paper and the model.py file.



### Training/Testing Schemes.

In this paper, we adopt 3 Training Schemes and 4 Test Schemes.

![](https://github.com/caoxingdong/adTMO/blob/master/models/imgs/train.png?raw=true)

![](https://github.com/caoxingdong/adTMO/blob/master/models/imgs/test.png?raw=true)

More details about Training/Testing Schemes can be accessed through the paper.

