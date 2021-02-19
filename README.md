# Condition-GAN-pytorch
Pytorch Implementation of paper "Conditional Generative Adversarial Nets" and "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets" 
## Training
```bash
# training using mnist
python3 src/main.py --config configs/mnist.txt --train True
# evaluation 
python3 src/main.py --config configs/mnist_eval.txt --eval True
```
