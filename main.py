import numpy as np
# from playsound import playsound
from scipy.io.wavfile import read


def load_handel_music() -> np.ndarray:
    """
    Loads music by Handel.
    """

    # Load music file.
    a = read("data/handel.wav")
    np.array(a[1], dtype=float)

    # # Play music.
    # playsound("/Homework2/Data/handel.wav")


def main() -> None:
    """
    Main program.
    """
    
    # Part 1
    load_handel_music()


if __name__ == '__main__':
    main()