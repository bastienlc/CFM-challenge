import numpy as np

EPSILON = 1e-5


def compute_sequence_features(rows):
    result = np.zeros((89))
    offset = 0

    for k in range(11):
        result[offset] = np.mean(rows[:, k])
        result[offset + 1] = np.min(rows[:, k])
        result[offset + 2] = np.max(rows[:, k])
        result[offset + 3] = np.std(rows[:, k])
        result[offset + 4] = np.median(rows[:, k])
        offset += 5

    local_features = [
        rows[:, 7] - rows[:, 6],
        np.abs(rows[:, 4] - rows[:, 5]) < EPSILON,
        np.abs(rows[:, 4] - rows[:, 6]) < EPSILON,
        rows[7] + rows[8],
    ]

    for feature in local_features:
        result[offset] = np.mean(feature)
        result[offset + 1] = np.min(feature)
        result[offset + 2] = np.max(feature)
        result[offset + 3] = np.std(feature)
        result[offset + 4] = np.median(feature)
        offset += 5

    global_features = [
        len(np.unique(rows[:, 1])),
        np.count_nonzero(rows[:, 0] == 0),
        np.count_nonzero(rows[:, 0] == 1),
        np.count_nonzero(rows[:, 0] == 2),
        np.count_nonzero(rows[:, 0] == 3),
        np.count_nonzero(rows[:, 0] == 4),
        np.count_nonzero(rows[:, 0] == 5),
        np.count_nonzero(rows[:, 2] == 0),
        np.count_nonzero(rows[:, 2] == 1),
        np.count_nonzero(rows[:, 3] == 0),
        np.count_nonzero(rows[:, 3] == 1),
        np.count_nonzero(rows[:, 9] == 0),
        np.count_nonzero(rows[:, 9] == 1),
        np.sum(rows[:, 10]),
    ]

    for feature in global_features:
        result[offset] = feature
        offset += 1

    return result
