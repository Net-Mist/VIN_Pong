"""
highly inspire from the bottleneck cached technique in :
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
"""
import random
import numpy as np

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

possible_actions = ['0', '3', '4']
label_to_index = [0, -1, -1, 1, 2]

# BE CAREFUL : the neural network lean the indices. Need to be mapped to action at the end
index_to_action = [0, 3, 4]


def load_data(data_file: str, testing_percentage: float, validation_percentage: float) -> {}:
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    :param data_file: String path to the .npy file containing information about images.
    :param testing_percentage: Integer percentage of the images to reserve for tests.
    :param validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """

    # load the npy archive
    dataset_info = np.load(data_file)

    result = {}
    for i in range(len(possible_actions)):
        result[i] = {
            'training': [],
            'testing': [],
            'validation': [],
        }

    for data in dataset_info:  # data = [simulation_id, frame_nb, action, y_player, y_op, x_b, y_b, x_b_old, y_b_old]
        random_percentage = random.randrange(100)
        action = data[2]
        keep_data = data[3:]
        if random_percentage < validation_percentage:
            result[action]['validation'].append(keep_data)
        elif random_percentage < (testing_percentage + validation_percentage):
            result[action]['testing'].append(keep_data)
        else:
            result[action]['training'].append(keep_data)

    training_nb = 0
    testing_nb = 0
    validation_nb = 0
    for i in result:
        training_nb += len(result[i]['training'])
        testing_nb += len(result[i]['testing'])
        validation_nb += len(result[i]['validation'])
    print(str(training_nb) + ' training images, ' + str(testing_nb) + ' testing images ' + str(
        validation_nb) + ' validation images.')
    return result


def get_random_cached_images(image_lists: {}, how_many: int, category: str, block_number:int) -> ([], [], []):
    """
    Args:
        image_lists: Dictionary of training images for each label.
        how_many: The number of bottleneck values to return.
        category: Name string of which set to pull from - training, testing, or validation.
    Returns:
        List of bottleneck arrays and their corresponding ground truths.
    """

    y_player = []
    other = []
    labels = []
    for _ in range(how_many):
        # randomly chose the class among all
        label_index = random.randrange(3)

        # randomly chose a state among the chosen class
        state_index = random.randrange(len(image_lists[label_index][category]))
        state = image_lists[label_index][category][state_index]

        # here we don't need to to this because we're using tf.nn.sparse_softmax_cross_entropy_with_logits
        # ground_truth = np.zeros(class_count, dtype=np.float32)
        # ground_truth[label_index] = 1.0
        y_player_vector = np.zeros(block_number)
        y_player_vector[state[0]] = 1
        other.append(state[1:])
        y_player.append(y_player_vector)
        labels.append(label_index)

    return y_player, other, labels
