import numpy as np


def generate_transition_sample(sample_a, sample_b, mixing_ratio):
    """
    Generate a transition sample by blending two activity samples.
    """
    # Note: UCI-HAR uses [Channels, Timesteps] layout, so transpose to [Timesteps, Channels] before calling this function.
    T = sample_a.shape[0]

    split_point = int(T * (1 - mixing_ratio))

    transition_sample = np.zeros_like(sample_a)

    transition_sample[:split_point, :] = sample_a[:split_point, :]

    transition_sample[split_point:, :] = sample_b[:T - split_point, :]

    return transition_sample


def augment_with_transitions(X, y, label_a, label_b, mixing_ratio=0.2, aug_ratio=0.1):
    """
    Augment a dataset by generating synthetic transition samples between two activity labels.
    """
    indices_a = np.where(y == label_a)[0]
    indices_b = np.where(y == label_b)[0]

    if len(indices_a) == 0 or len(indices_b) == 0:
        raise ValueError(f"label_a={label_a} or label_b={label_b} not found in y")

    num_aug = int(len(X) * aug_ratio)

    new_samples = []
    new_labels = []

    for _ in range(num_aug):
        idx_a = np.random.choice(indices_a)
        idx_b = np.random.choice(indices_b)

        combined = generate_transition_sample(X[idx_a], X[idx_b], mixing_ratio)

        new_samples.append(combined)
        new_labels.append(label_a)

    X_augmented = np.concatenate([X, np.array(new_samples)], axis=0)
    y_augmented = np.concatenate([y, np.array(new_labels)], axis=0)

    p = np.random.permutation(len(X_augmented))
    return X_augmented[p], y_augmented[p]