import gym
from vin_atari import *
import tensorflow as tf
import numpy as np
import imageProcessor
from PIL import Image

ATARI_SCREEN_WIDTH = 210
ATARI_SCREEN_HEIGHT = 160
ATARI_SCREEN_GAME_WIDTH = 128
ATARI_SCREEN_GAME_HEIGHT = 136
K_REC = 40
NUM_ACTIONS = 18

labels = [16, 14, 13, 2, 6, 0, 4, 17, 1, 7, 12, 11, 5, 9, 15, 3, 10, 8]


def main():
    print("Init OpenAI part")
    env = gym.make('BankHeist-v0')
    observation = env.reset()
    # print(observation.shape) #(250, 160, 3)
    with tf.Graph().as_default():
        print('Create network placeholder')
        images_placeholder1, images_placeholder2, maps_placeholder, positions_placeholder = \
            placeholders_openai(ATARI_SCREEN_WIDTH, ATARI_SCREEN_HEIGHT, ATARI_SCREEN_GAME_WIDTH,
                                ATARI_SCREEN_GAME_HEIGHT)

        print('Create the inference part')
        logits = inference(images_placeholder1, images_placeholder2, maps_placeholder, positions_placeholder,
                           NUM_ACTIONS, K_REC)

        # init = tf.global_variables_initializer()
        sess = tf.Session()
        # sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, "../TensorFlow-VIN/model.ckpt-1499")

        for _ in range(1000):
            env.render()

            image2 = Image.fromarray(observation).crop((12, 67, 148, 195))
            # image2 = observation[148:196, 12:68, :]
            game_map, car_position, car_density, ghost_positions, police_car_density = imageProcessor.process(np.array(image2))

            # merge all data
            merge = np.zeros((16, 34, 4))
            merge[:, :, 0] = game_map

            for i in ghost_positions:
                merge[i[0], i[1], 1] = 1

            for i in police_car_density:
                merge[i[0], i[1], 2] = i[2]

            for i in car_density:
                merge[i[0], i[1], 3] = i[2]

            position = np.zeros((16, 34))
            position[car_position[0], car_position[1]] = 1

            l = sess.run([logits], feed_dict={
                                              maps_placeholder: [merge],
                                              positions_placeholder: [position]})
            # observation, reward, done, info = env.step(labels[np.argmax(l)])
            observation, reward, done, info = env.step(np.argmax(l))
            print(l)
            if done:
                print("Episode finished")
                break


if __name__ == '__main__':
    main()
