import time
from image_preprocess import *

block_size = 8  # Should be a divisor of 160
number_blocks = int(160 / block_size)


def print_grid(ball_position_distribution, player_position_distribution, opponent_position_distribution):
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


start_time = time.time()

# load the image
image = Image.open(sys.argv[1])
y_player, y_opponent, x_b, y_b, player_position_distribution, opponent_position_distribution, ball_position_distribution = process(
    image)
duration = time.time() - start_time
print(duration)
print(y_player, y_opponent, x_b, y_b)
print_grid(ball_position_distribution, player_position_distribution, opponent_position_distribution)
