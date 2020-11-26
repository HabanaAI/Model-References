import time
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation


class VideoManager(object):
    def __init__(self, figsize=None, table_size=(1, 1), interval=100, repeat=True):
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.pause = True
        n_rows, n_cols = table_size
        self.spec = gridspec.GridSpec(nrows=n_rows, ncols=n_cols, figure=self.fig)
        self.interval = interval
        self.repeat = repeat
        self.videos = []
        self.images_to_play = []
        self.axis = []
        self.number_of_images = None

    def add_images(self, images, rows_from=None, rows_to=None, cols_from=None, cols_to=None, cmaps='viridis',
                   origin=None, title=None, alphas=None):
        if not alphas:
            images = [images]
            alphas = [1.0]
        if type(cmaps) is not list:
            cmaps = [cmaps] * len(images)
        for list_of_images in images:
            if not self.number_of_images:
                self.number_of_images = len(list_of_images)
            assert self.number_of_images == len(list_of_images)
        ax = plt.subplot(self.spec[rows_from: rows_to, cols_from:cols_to])
        if title:
            ax.title.set_text(title)
        images_to_play = []
        for i in range(self.number_of_images):
            frame_images = []
            for j, list_of_images in enumerate(images):
                frame_images.append(ax.imshow(list_of_images[i], cmap=cmaps[j], origin=origin, alpha=alphas[j]))
            images_to_play.append(frame_images)
        self.images_to_play.append(images_to_play)

    def play(self, save_to=None, fps=10):
        for imgs in self.images_to_play:
            self.videos.append(animation.ArtistAnimation(self.fig, imgs, blit=True, interval=self.interval,
                                                         repeat=self.repeat))
        self.pause = False
        if save_to and len(self.videos) == 1:
            self.videos[0].save(save_to, fps=fps)
        plt.show()

    def on_press(self, event):
        if event.key == ' ':
            self.pause ^= True
            for video in self.videos:
                if self.pause:
                    video.event_source.stop()
                else:
                    video.event_source.start()
        if event.key == 'down':
            if self.interval < 2000:
                for video in self.videos:
                    video.event_source.interval += 100
                self.interval += 100

        if event.key == 'up':
            if self.interval > 100:
                for video in self.videos:
                    video.event_source.interval -= 100
                self.interval -= 100

        if event.key == 'right' and self.pause:
            for video in self.videos:
                video.event_source.start()
            time.sleep(self.interval)
            for video in self.videos:
                video.event_source.stop()
