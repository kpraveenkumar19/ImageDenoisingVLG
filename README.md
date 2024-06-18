# Image Denoising 

## Files

- `imagedenoising.ipynb`: Jupyter notebook containing the implementation of the image denoising autoencoder.
- `autoencoder.h5`: Trained model file for the autoencoder.


## Introduction

This project demonstrates the implementation and evaluation of image denoising using deep convolutional autoencoders. The primary metrics for evaluation are the Peak Signal-to-Noise Ratio (PSNR) and the Structural Similarity Index (SSIM). We trained and compared two autoencoder models to identify the superior one.

## Architecture Specifications

The autoencoder architecture used in this project includes:

- **Input Layer**: Shape (256, 256, 3)

- **Encoder**:
  - Conv2D with 32 filters
  - MaxPooling2D
  - Conv2D with 64 filters
  - MaxPooling2D
  - Conv2D with 128 filters
  - MaxPooling2D

- **Decoder**:
  - Conv2DTranspose with 128 filters
  - Conv2DTranspose with 64 filters
  - Conv2DTranspose with 32 filters
  - Conv2D with 3 filters (output layer)

The model was compiled with the Adam optimizer and Mean Squared Error loss function. The best model achieved a PSNR of 27.95 on the test set.

## Project Details

### Data Preparation

The dataset includes high-quality and low-quality images. The images were divided into training and testing sets, and further processed into patches of size 256x256 for training the autoencoder.

### Visualization of Training and Testing Sets

We visualized the training and testing sets to inspect the quality of images and ensure proper loading.

### Performance Metrics Calculation

The performance of the autoencoder was evaluated using PSNR and MSE metrics.

### Autoencoder Training

The autoencoder was trained using the preprocessed image patches.

## Denoising and Evaluation

The trained model was used to denoise the test images, and its performance was evaluated using the PSNR and SSIM metrics.

## Results

The autoencoder successfully denoised the images, achieving an average PSNR of 27.95 on the test set. The performance metrics indicate that the model can effectively reduce noise and enhance image quality.

### Methods for Improvement

- **Data Augmentation**: Increasing the variety of training data through augmentation techniques can improve model robustness.
- **Hyperparameter Tuning**: Experimenting with different hyperparameters, such as learning rates, batch sizes, and the number of layers, may yield better performance.
- **Advanced Architectures**: Implementing more sophisticated architectures like U-Net or using pre-trained models for transfer learning could enhance results.
- **Post-processing**: Applying post-processing techniques to the denoised images can further improve quality.

### Conclusion

The project demonstrates the effectiveness of convolutional autoencoders for image denoising. Future work will focus on optimizing the model and exploring more advanced techniques to further enhance denoising performance.

### References

- [Image Denoising Using Deep Learning](#)
- [Image Denoising Using Deep Learning on Medium](#)
- [Image Denoising Using Deep Learning on Towards AI](#)

