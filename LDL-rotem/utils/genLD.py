import numpy as np

# Generate label distribution
def gen_label_distribution(label, sigma, loss, num_classes):

    label_set = np.array(range(num_classes))

    if loss == 'klloss':
        ld_num = len(label_set)
        dif_age = np.tile(label_set.reshape(ld_num, 1), (1, len(label))) - np.tile(label, (ld_num, 1))
        ld = 1.0 / np.tile(np.sqrt(2.0 * np.pi) * sigma, (ld_num, 1)) * np.exp(-1.0 * np.power(dif_age, 2) / np.tile(2.0 * np.power(sigma, 2), (ld_num, 1)))
        ld = ld / np.sum(ld, 0)

        return ld.transpose()
