import numpy as np
from PIL import Image
import sys

# Size of the bare = 16
# Size of the ball = 4*2

block_size = 8  # Should be a divisor of 160
number_blocks = int(160 / block_size)

mean_ball = np.array([255, 228, 197])
mean_player = np.array([92, 185, 94])
mean_opponent = np.array([212, 130, 74])
mean_background = np.array([143, 72, 16])

player_position_distribution = np.zeros(number_blocks, dtype='int')
opponent_position_distribution = np.zeros(number_blocks, dtype='int')
ball_position_distribution = np.zeros((number_blocks, number_blocks), dtype='int')

player_position = np.zeros(number_blocks, dtype='bool')
opponent_position = np.zeros(number_blocks, dtype='bool')
ball_position = np.zeros((number_blocks, number_blocks), dtype='bool')


def distance(color1, color2):
    return np.sum(np.absolute(color1 - color2))


def compute_distribution_1d(pixel_array, special_color):
    """
    :param pixel_array: np.array of shape (block_size, 3)
    :param special_color:
    :return:
    """
    count = 0
    for i in pixel_array:
        if distance(i, special_color) < distance(i, mean_background):
            count += 1
    return count


def fill_position_distribution_1d(array_to_fill, initial_array, special_color):
    for i in range(number_blocks):
        first_pixel = i * block_size
        last_pixel = (i + 1) * block_size - 1
        array_to_fill[i] = compute_distribution_1d(initial_array[first_pixel:last_pixel + 1], special_color)
    return


def compute_distribution_2d(pixel_array, special_color):
    """
    :param pixel_array: np.array of shape (block_size, block_size, 3)
    :param special_color: np.array of shape (3)
    :return:
    """
    count = 0
    for i in range(block_size):
        for j in range(block_size):
            if distance(pixel_array[i, j, :], special_color) < distance(pixel_array[i, j, :], mean_background):
                count += 1
    return count


def fill_position_distribution_2d(array_to_fill, initial_array, special_color):
    for i in range(number_blocks):
        for j in range(number_blocks):
            array_to_fill[i, j] = compute_distribution_2d(initial_array[i * block_size:(i + 1) * block_size,
                                                          j * block_size:(j + 1) * block_size],
                                                          special_color)
    return


def print_grid():
    for i in range(number_blocks):
        for j in range(number_blocks):
            if ball_position_distribution[i, j] != 0:
                print(ball_position_distribution[i, j], end='')
            elif j == 0 and opponent_position_distribution[i] != 0:
                print(opponent_position_distribution[i], end='')
            elif j == number_blocks - 1 and player_position_distribution[i] != 0:
                print(player_position_distribution[i], end='')
            else:
                print(".", end='')
        print()


# load the image
image = Image.open(sys.argv[1])
ball_area = image.crop((0, 34, 160, 194))  # Size : 160 * 160
opponent_area = image.crop((17, 34, 18, 194))  # Size : 1 * 160
player_area = image.crop((141, 34, 142, 194))  # Size : 1 * 160

# Find the positions distributions
fill_position_distribution_1d(player_position_distribution, np.array(player_area)[:, 0, :], mean_player)
fill_position_distribution_1d(opponent_position_distribution, np.array(opponent_area)[:, 0, :], mean_opponent)
fill_position_distribution_2d(ball_position_distribution, np.array(ball_area), mean_ball)

# Find the positions
player_position[np.argmax(player_position_distribution)] = 1
opponent_position[np.argmax(opponent_position_distribution)] = 1
ball_position[np.argmax(np.sum(ball_position_distribution, 1)), np.argmax(np.sum(ball_position_distribution, 0))] = 1

print(player_position)
print(opponent_position)
print(ball_position)
