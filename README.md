This is a simple project applying a Gabor Filter Bank and SVM classifier to the MNIST dataset, output formatted for a Kaggle competition (https://www.kaggle.com/c/digit-recognizer/overview).

The Gabor filters used here for dimensionality reduction do not seem to do a good job, indeed passing the raw data to the SVC seems to have better performance in cross-validation scoring over the training data. There are a number of potential reasons for this to be explored in future work.

Future Work:
The images are small (28x28) and I couldn't make the filters any smaller than ~(13x13). I don't have a good reason to justify the intuition, but I feel the filters should be smaller.

The range of filter parameters may not be optimal, I only tried a small range of values for theta (orientation) and sigma (spatial frequency).

The algorithm may not be appropriate for recognising handwritten digits. I would need to plot more samples but I imagine that one of the main ways a representation of a digit will change is a slight rotation or shift of the angles. This would change which filters matched and so have a different feature representation.
