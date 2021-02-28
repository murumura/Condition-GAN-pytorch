# Condition-GAN-pytorch
Pytorch Implementation of paper "Conditional Generative Adversarial Nets" and "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets" 
## Training
### MNIST
```bash
# training using mnist
python3 src/main.py --config configs/cgan_mnist.txt --train True
# evaluation 
python3 src/main.py --config configs/cgan_mnist_eval.txt --eval True
# alternatively training using infogan:
python3 src/main.py --config configs/infogan_mnist.txt --train True

```
#### Result

### Fashion-MNIST
```bash
# training using mnist
python3 src/main.py --config configs/cgan_fashion_mnist.txt --train True
# evaluation 
python3 src/main.py --config configs/cgan_fashion_mnist_eval.txt --eval True
```
#### Result
<center class="half">
    <img src="data/asset/fashion_mnist/loss.gif" width="200"/><img src="data/asset/fashion_mnist/synsethesis.gif" width="200"/>
</center>