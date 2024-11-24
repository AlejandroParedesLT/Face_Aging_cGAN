import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm

# Load InceptionV3 for FID and Inception Score calculations
def get_inception_model():
    """
    Load the InceptionV3 model pre-trained on ImageNet.
    The model outputs features from the penultimate layer (pooling layer).
    """
    base_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    return base_model


def preprocess_images(images):
    """
    Preprocess images for InceptionV3.
    Normalize to [-1, 1] and resize to 299x299.
    """
    images = tf.image.resize(images, (299, 299))
    images = tf.keras.applications.inception_v3.preprocess_input(images)
    return images


def calculate_fid(real_images, fake_images, model):
    """
    Calculate Fréchet Inception Distance (FID).
    :param real_images: Numpy array of real images [N, H, W, C]
    :param fake_images: Numpy array of fake images [N, H, W, C]
    :param model: Pre-trained InceptionV3 model
    :return: FID score
    """
    real_images = preprocess_images(real_images)
    fake_images = preprocess_images(fake_images)

    # Extract features
    real_features = model.predict(real_images, batch_size=32)
    fake_features = model.predict(fake_images, batch_size=32)

    # Calculate mean and covariance of features
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    # Compute FID
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_inception_score(images, model, splits=10):
    """
    Calculate Inception Score.
    :param images: Numpy array of images [N, H, W, C]
    :param model: Pre-trained InceptionV3 model
    :param splits: Number of splits for the score calculation
    :return: Tuple (mean, std) of the Inception Score
    """
    images = preprocess_images(images)
    preds = model.predict(images, batch_size=32)

    # Softmax predictions
    preds = tf.nn.softmax(preds).numpy()
    scores = []

    # Split predictions into `splits`
    split_size = len(preds) // splits
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        kl_div = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))
    return np.mean(scores), np.std(scores)


def load_images_from_dir(directory, target_size=(299, 299)):
    """
    Load all images from a directory and resize them to the target size.
    :param directory: Path to the directory containing images
    :param target_size: Target size of the images (height, width)
    :return: Numpy array of images
    """
    import os
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img)
            images.append(img)
    print(images)
    return np.array(images)


def evaluate_metrics(real_image_dir, fake_image_dir):
    """
    Load real and fake images, then calculate FID and Inception Score.
    :param real_image_dir: Path to directory with real images
    :param fake_image_dir: Path to directory with fake images
    """
    # Load real and fake images
    real_images = load_images_from_dir(real_image_dir)
    fake_images = load_images_from_dir(fake_image_dir)

    # Load InceptionV3 model
    model = get_inception_model()

    print("Calculating FID...")
    fid_score = calculate_fid(real_images, fake_images, model)
    print(f"FID Score: {fid_score}")

    print("Calculating Inception Score...")
    inception_mean, inception_std = calculate_inception_score(fake_images, model)
    print(f"Inception Score: Mean = {inception_mean}, Std = {inception_std}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FID and Inception Score.")
    parser.add_argument('--real_image_dir', type=str, required=True, help="Path to real images directory.")
    parser.add_argument('--fake_image_dir', type=str, required=True, help="Path to generated images directory.")

    args = parser.parse_args()

    evaluate_metrics(args.real_image_dir, args.fake_image_dir)