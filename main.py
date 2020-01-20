import os
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.io.wavfile import read

import utils

HANDEL_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/gabor-transforms/handel.wav'
PLOTS_FOLDER = 'plots'
COLOR1 = '#3F4F8C'


def load_handel_music() -> np.ndarray:
    """
    Loads music by Handel.
    """

    # Load Handel music file.
    handel_path = utils.download_remote_data_file(HANDEL_URL)
    # sample_rate is measurements per second.
    (sample_rate, handel_data) = read(handel_path)

    # 65536 = 2^16. 
    # Samples in a wav file have 16 bits, so we scale the amplitudes to be 
    # between 0 and 1.
    handel_data = handel_data/65536

    return (sample_rate, handel_data)


def plot_handel_data(sample_rate: float, data: np.ndarray, dirname: str, 
    filename: str) -> None:
    """
    Plots the amplitude of the Handel music over time.
    """

    print(f'Plotting Handel music data...')

    path = os.path.join(dirname, filename)

    t = np.arange(0, len(data))/sample_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=data,
            mode='lines',
            line_color=COLOR1,
            line_width=3,
        )
    )
    fig.update_layout(
        title_text='Handel music data',
        xaxis_title_text='Time in seconds',
        yaxis_title_text='Amplitude',
    )
    pio.write_html(fig, path)


def plot_spectrograms(spectrograms: List[np.ndarray], plot_x: List[np.ndarray], 
    plot_y: List[np.ndarray], plot_titles: List[str], dirname: str, 
    filename: str) -> None:
    """
    Plots a list of spectrograms.
    """

    print(f'Plotting spectrograms...')

    path = os.path.join(dirname, filename)

    rows = len(spectrograms)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=plot_titles)
    for row in range(rows):
        fig.add_trace(
            go.Heatmap(z=spectrograms[row],
                x=plot_x[row],
                y=plot_y[row],
                coloraxis='coloraxis',
            ),
            col=1,
            row=row+1,
        )
    fig.update_yaxes(
        title_text='Frequency (omega)',
    )
    fig.update_xaxes(
        title_text='Time (t)',
    )
    fig.update_layout(
        coloraxis_colorscale='Viridis',
    )
    pio.write_html(fig, path)


def try_different_gabor_widths(sample_rate: float, 
    data: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], 
    List[np.ndarray], List[str]]:
    """
    Filters the temporal data using Gaussian Gabor filter with different
    widths, and transforms the result using FFT.
    """

    max_time = len(data)/sample_rate

    # Subsample the data.
    num_samples = 200
    sampled_data = data[::len(data)//num_samples]

    # Time steps.
    b_list = np.linspace(0, max_time, len(sampled_data))

    # Gaussian filter standard deviations.
    sigma_list = [0.1, 0.3, 0.7]

    spectrograms = []
    plot_x = []
    plot_y = []
    plot_titles = []

    # For each Gaussian filter width:
    for sigma in sigma_list:
        spectrogram = np.empty((len(b_list), len(b_list)))
        a = 1/(2*sigma**2)
        # For each time step, slide the Gabor filter so that it's centered at 
        # the desired time, apply it to the function in time domain, and 
        # transform the result using FFT.
        for (j, b) in enumerate(b_list):
            g = np.exp(-a*(b_list-b)**2)
            ug = sampled_data * g
            ugt = np.fft.fftshift(np.fft.fft(ug))
            spectrogram[:, j] = utils.normalize(ugt)

        spectrograms.append(spectrogram)

        plot_x.append(b_list)

        omega_points = np.linspace(-num_samples/2, num_samples/2, num_samples+1)[0:-1]
        omega_shifted = (2 * np.pi)/max_time * omega_points
        plot_y.append(omega_shifted)

        plot_titles.append(f'Spectrogram using Gabor Gaussian filter with standard deviation = {sigma}')

    return (spectrograms, plot_x, plot_y, plot_titles)


def try_different_gabor_timesteps(sample_rate: float, 
    data: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], 
    List[np.ndarray], List[str]]:
    """
    Filters the temporal data using a Gaussian Gabor filter by sliding it
    with different timesteps, and transforms the result using FFT.
    """

    max_time = len(data)/sample_rate

    num_samples_list = [100, 400, 800]

    spectrograms = []
    plot_x = []
    plot_y = []
    plot_titles = []

    for num_samples in num_samples_list:
        # Subsample the data.
        sampled_data = data[::len(data)//num_samples]

        # Time steps.
        b_list = np.linspace(0, max_time, len(sampled_data))

        # Gaussian filter standard deviations.
        sigma = 0.1

        spectrogram = np.empty((len(b_list), len(b_list)))

        a = 1/(2*sigma**2)
        # For each time step, slide the Gabor filter so that it's centered at 
        # the desired time, apply it to the function in time domain, and 
        # transform the result using FFT.
        for (j, b) in enumerate(b_list):
            g = np.exp(-a*(b_list-b)**2)
            ug = sampled_data * g
            ugt = np.fft.fftshift(np.fft.fft(ug))
            spectrogram[:, j] = utils.normalize(ugt)

        spectrograms.append(spectrogram)

        plot_x.append(b_list)
        omega_points = np.linspace(-num_samples/2, num_samples/2, num_samples+1)[0:-1]
        omega_shifted = (2 * np.pi)/max_time * omega_points
        plot_y.append(omega_shifted)

        plot_titles.append(f'Spectrogram using Gabor Gaussian filter and {num_samples} time steps')

    return (spectrograms, plot_x, plot_y, plot_titles)


def try_different_gabor_functions(sample_rate: float, 
    data: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], 
    List[np.ndarray], List[str]]:
    """
    Filters the temporal data using different Gabor filters, and transforms 
    the result using FFT.
    """

    max_time = len(data)/sample_rate

    # Subsample the data.
    num_samples = 200
    sampled_data = data[::len(data)//num_samples]

    # Time steps.
    b_list = np.linspace(0, max_time, len(sampled_data))

    # Filter 1: Gaussian filter.
    sigma = 0.1
    a = 1/(2*sigma**2)
    gaussian = lambda b: np.exp(-a*(b_list-b)**2)

    # Filter 2: Box function.
    width = 1
    box_function = lambda b: np.heaviside(width/2-np.abs(b_list-b), 1)

    # Filter 3: Mexican hat function.
    sigma2 = 0.1
    mexican_hat_function = lambda b: (1-((b_list-b)/sigma2)**2) * np.exp(-(b_list-b)**2/(2*sigma2**2))

    # List of Gabor filters.
    g_list = [
        gaussian,
        box_function,
        mexican_hat_function
    ]
    filter_name_list = ['Gaussian', 'Box function', 'Mexican hat']

    spectrograms = []
    plot_x = []
    plot_y = []
    plot_titles = []

    for (i, g) in enumerate(g_list):
        spectrogram = np.empty((len(b_list), len(b_list)))

        # For each time step, slide the Gabor filter so that it's centered at 
        # the desired time, apply it to the function in time domain, and 
        # transform the result using FFT.
        for (j, b) in enumerate(b_list):
            ug = sampled_data * g(b)
            ugt = np.fft.fftshift(np.fft.fft(ug))
            spectrogram[:, j] = utils.normalize(ugt)

        spectrograms.append(spectrogram)

        plot_x.append(b_list)

        omega_points = np.linspace(-num_samples/2, num_samples/2, num_samples+1)[0:-1]
        omega_shifted = (2 * np.pi)/max_time * omega_points
        plot_y.append(omega_shifted)

        plot_titles.append(f'Spectrogram using {filter_name_list[i]} filter')

    return (spectrograms, plot_x, plot_y, plot_titles)


def main() -> None:
    """
    Main program.
    """
        
    # Part 1
    (sample_rate, handel_data) = load_handel_music()
    plots_dir_path = utils.find_or_create_dir(PLOTS_FOLDER)
    plot_handel_data(sample_rate, handel_data, plots_dir_path, '1_handel_data.html')
    (spectrograms, plot_x, plot_y, 
        plot_titles) = try_different_gabor_widths(sample_rate, handel_data)
    plot_spectrograms(spectrograms, plot_x, plot_y, plot_titles, 
        plots_dir_path, '2_spectrograms_different_widths.html')
    (spectrograms, plot_x, plot_y, 
        plot_titles) = try_different_gabor_timesteps(sample_rate, handel_data)
    plot_spectrograms(spectrograms, plot_x, plot_y, plot_titles, 
        plots_dir_path, '3_spectrograms_different_timesteps.html')
    (spectrograms, plot_x, plot_y, 
        plot_titles) = try_different_gabor_functions(sample_rate, handel_data)
    plot_spectrograms(spectrograms, plot_x, plot_y, plot_titles, 
        plots_dir_path, '4_spectrograms_gabor_functions.html')


if __name__ == '__main__':
    main()
