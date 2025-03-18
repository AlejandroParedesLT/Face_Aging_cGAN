# Alejandro Paredes La Torre, Face Aging with cGAN

## 1. Introduction
This repository implements face aging using **Identity-Preserved Conditional Generative Adversarial Networks (IPCGAN)**, which allows for age transformations of facial images while preserving the identity of the subject. The goal of this model is to generate realistic, age-modified faces using a conditional GAN framework, ensuring that the generated face corresponds to a specific age group while maintaining the identity of the individual.

The model includes a ResNet-based generator, a PatchGAN discriminator, and an AlexNet-based feature extraction network for age classification. It uses a dataset of celebrity images from the **IMDB-WIKI** dataset and evaluates the results using metrics like **Fréchet Inception Distance (FID)** and **Inception Score (IS)**.

![alt text](https://github.com/AlejandroParedesLT/Face_Aging_cGAN/blob/main/grouped_generated_images/17_Dakota_Johnson_0009.jpg_1x5.jpg?raw=true)
![alt text](https://github.com/AlejandroParedesLT/Face_Aging_cGAN/blob/main/grouped_generated_images/15_Chris_Colfer_0009.jpg_1x5.jpg?raw=true)
## 2. Installation

To get started, first clone this repository and install the necessary Python dependencies. The required dependencies are listed in `requirements.txt`.

### 2.1. Install Dependencies
Ensure you have Python 3.6 or above. Install the dependencies using the following command:

```shell
pip install -r requirements.txt
```

**Required dependencies include**:

- `tensorflow-gpu==1.4.1`
- `scipy==1.0.0`
- `opencv-python==3.3.0.10`
- `numpy==1.11.0`
- `Pillow==5.1.0`

### 2.2. Set Up GPU (Optional)
If you're working with a GPU, ensure you have **CUDA** and **cuDNN** installed. This will allow you to take advantage of GPU acceleration during training.

## 3. Download Datasets

This implementation uses the **IMDB-WIKI** dataset, which consists of over 500,000 celebrity face images labeled with age and gender information.

You can obtain the dataset from [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), or use a preprocessed version if you prefer.

### 3.1. Dataset Structure

The dataset is split into 5 age groups:

- Group 0: 0-20
- Group 1: 20-30
- Group 2: 30-40
- Group 3: 40-50
- Group 4: 50+

## 4. Testing with Saved Models

To test the model on pre-trained weights:

### 4.1. Download Pre-trained Model

Download the trained **face aging model** from the following link:

[Download Trained Model](https://1drv.ms/u/s!AlUWwwOcwDWobCqmuFyKGIt4qaA)

Then, place the model files in the `checkpoints/0_conv5_lsgan_transfer_g75_0.5f-4_a30` directory.

### 4.2. Test on Images

Test images are in the `images/test` directory, and some training images that belong to the 11-20 age group are in `images/train`.

To generate aged faces from the test images, run the following script:

```shell
python test.py
```

## 5. Training from Scratch

To train the model from scratch, first download the pre-trained **AlexNet model** and **age classification model** from the following links:

- [Download Pre-trained AlexNet Model](https://1drv.ms/u/s!AlUWwwOcwDWobkptownyu5fjlfU)
- [Download Pre-trained Age Classification Model](https://1drv.ms/f/s!AlUWwwOcwDWocX-Z0IJft_VbcoQ)

After downloading, unzip these files and place the model files in `checkpoints/pre_trained`.

### 5.1. Training the Model

To train the model with the downloaded pre-trained weights, use the following command:

```shell
python age_lsgan_transfer.py \
  --gan_loss_weight=75 \
  --fea_loss_weight=0.5e-4 \
  --age_loss_weight=30 \
  --fea_layer_name=conv5 \
  --checkpoint_dir=./checkpoints/age/0_conv5_lsgan_transfer_g75_0.5f-4_a30 \
  --sample_dir=age/0_conv5_lsgan_transfer_g75_0.5f-4_a30 
```

This will begin the training process, generating age-modified faces and saving the results to the specified directory (`sample_dir`).

Alternatively, you can use the shell script to train the model:

```shell
sh age_lsgan_transfer.py
```

### 5.2. Training Hyperparameters

- `gan_loss_weight`: The weight of the adversarial loss.
- `fea_loss_weight`: The weight of the feature loss.
- `age_loss_weight`: The weight of the age loss.
- `fea_layer_name`: The name of the feature extraction layer (e.g., `conv5` for AlexNet).
- `checkpoint_dir`: Directory to save model checkpoints.
- `sample_dir`: Directory to save generated samples.

## 6. Model Evaluation

After training, the model can be evaluated using two key metrics:

- **Fréchet Inception Distance (FID)**: Measures the distance between feature distributions of real and generated images. A lower FID indicates more realistic images.
- **Inception Score (IS)**: Evaluates the diversity and quality of generated images. A higher IS suggests better image quality and diversity.

### 6.1. Evaluation Results

- **FID Score**: 63.14 (for a subset of 30 test images)
- **Inception Score**: Mean = 1.035, Std = 0.01

These results demonstrate the effectiveness of the model in generating realistic aged faces while preserving the identity of the individuals.

## 7. Conclusion

This implementation demonstrates the effectiveness of using **Identity-Preserved Conditional GANs** (IPCGANs) for controlled face aging. The model successfully transforms faces to specific age groups while retaining identity, providing a useful tool for applications such as age progression/regression in image processing. Future work could focus on improving image quality, handling more diverse datasets, and reducing noise in generated images.


## THis implementation was based on:
```code
@INPROCEEDINGS{wang2018face_aging, 
	author={Z. Wang and X. Tang, W. Luo and S. Gao}, 
	booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
	title={Face Aging with Identity-Preserved Conditional Generative Adversarial Networks}, 
	year={2018}
}
```
