import tensorflow as tf
import numpy as np
from scipy.linalg import sqrtm
#tf.compat.v1.enable_eager_execution()

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

    # Calculate number of steps
    steps_real = real_images.shape[0] // 32
    steps_fake = fake_images.shape[0] // 32

    # Extract features
    real_features = model.predict(real_images, batch_size=32, steps=steps_real)
    fake_features = model.predict(fake_images, batch_size=32, steps=steps_fake)

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
    steps_fake = images.shape[0] // 32

    # Model prediction
    preds = model.predict(images, batch_size=32, steps=steps_fake)
    
    # Apply softmax
    preds = tf.nn.softmax(preds).numpy()  # Ensure NumPy array

    scores = []

    # Split predictions into `splits`
    split_size = len(preds) // splits
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        kl_div = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))
    return np.mean(scores), np.std(scores)


def evaluate_metrics_training(source, samples):
    """
    Evaluate FID and Inception Score on source and generated samples.
    :param source: Numpy array of real images [N, H, W, C]
    :param samples: Numpy array of generated images [N, H, W, C]
    """
    # Ensure source and samples are numpy arrays with the correct shape
    if isinstance(source, np.ndarray) and isinstance(samples, np.ndarray):
        print(f"Source shape: {source.shape}, Samples shape: {samples.shape}")
    else:
        print("Input images must be numpy arrays.")
        return
    
    # Load InceptionV3 model
    model = get_inception_model()

    print("Calculating FID...")
    fid_score = calculate_fid(source, samples, model)
    print(f"FID Score: {fid_score}")

    print("Calculating Inception Score...")
    inception_mean, inception_std = calculate_inception_score(samples, model)
    print(f"Inception Score: Mean = {inception_mean}, Std = {inception_std}")


# Example usage
if __name__ == "__main__":
    # Replace these with your actual image data
    source_images = np.random.rand(32, 128, 128, 3)  # Replace with actual source images
    generated_images = np.random.rand(32, 128, 128, 3)  # Replace with generated samples

    evaluate_metrics_training(source_images, generated_images)
