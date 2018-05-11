import imageio
import glob

# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/path/to/movie.gif', images)