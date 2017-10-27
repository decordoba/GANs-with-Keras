from datetime import datetime
import numpy as np
from LumpyBgnd import lumpy_backround, create_lumps_pos_matrix
from click.termui import pause
from matplotlib import pyplot as plt


def get_current_time(time=True, date=False):
    now = datetime.now()
    s = ""
    if date:
        s += "{} ".format(now.date())
    if time:
        s += "{:02d}:{:02d}:{:02d}".format(now.hour, now.minute, now.second)
    return s.strip()


def generate_data(save_lumps_pos=False, show_images=False, pause_images=False,
                  discrete_centers=False):
    FIRST_SAMPLE_NUM = 0  # The 1st sample will have this number
    LAST_SAMPLE_NUM = 99  # The last sample will have this number
    num_samples = LAST_SAMPLE_NUM - FIRST_SAMPLE_NUM + 1
    print("Samples generated: " + str(num_samples))

    IMAGE_SIZE = [64, 64]
    MEAN_NUMBER_LUMPS = 200
    DC = 10
    LUMP_FUNCTION = 'GaussLmp'
    PARS = [1, 2] # [1, 10]

    DATASET_NAME = "lumps3"

    if show_images:
        plt.ion()

    # save or show data
    percent = 2
    split_distance = num_samples * percent // 100
    for i in range(FIRST_SAMPLE_NUM, LAST_SAMPLE_NUM + 1):
        matrix, num_lumps, pos_lumps = lumpy_backround(IMAGE_SIZE, MEAN_NUMBER_LUMPS, DC,
                                                       LUMP_FUNCTION, PARS,
                                                       discretize_lumps_positions=discrete_centers)

        # Save all lumpy_images
        try:
            y = np.concatenate((y, [matrix]))
        except NameError:
            y = np.array([matrix])

        # Only create matrix with lumps centers if we are going to save it
        if save_lumps_pos:
            pos_matrix = create_lumps_pos_matrix(dim=IMAGE_SIZE, lumps_pos=pos_lumps)
            # Save all matrices with lumps centers
            try:
                x = np.concatenate((x, [pos_matrix]))
            except NameError:
                x = np.array([pos_matrix])

        if show_images:
            if save_lumps_pos:
                fig = plt.figure(0)
                ax = fig.add_subplot(1, 2, 1)
                ax.imshow(pos_matrix)
                ax.set_yticks([])
                ax.set_xticks([])
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(matrix)
                ax.set_yticks([])
                ax.set_xticks([])
            else:
                plt.imshow(matrix)
                plt.yticks([])
                plt.xticks([])
            plt.pause(0.00001)
            if pause_images:
                s = input("Press ENTER to see the next image, or Q (q) to disable pause:  ")
                if len(s) > 0 and s[0].lower() == "q":
                    pause_images = False

        if (i + 1) % split_distance == 0:
            print("{}. {}% loaded".format(get_current_time(), (i + 1) * 100 // num_samples))

    if save_lumps_pos:
        data = (x, y)
    else:
        data = y
    np.save(DATASET_NAME, data)

    if show_images:
        plt.ioff()

if __name__ == "__main__":
    generate_data(save_lumps_pos=True, show_images=True, pause_images=True, discrete_centers=False)

    """
    HOW TO USE:
        save_lumps_pos: True, a lumps_position matrix is saved, False, only the generated lumpy
                        image is saved
        show_images: True, shows images of lumpy image (and lumps_position matrix), False, doesn't
        pause_images: True, pauses in every image, you have to type ENTER to proceed, False doesn't
        discrete_centers: only use discrete values (ints) for the centers of the lumps, False
                          the center of the lump is shared by the pixels depending on distance to
                          every adjacent pixel
    """