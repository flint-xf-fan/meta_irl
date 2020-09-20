import os

import gym
from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, path, filename):
    plt.figure(figsize=(frames[0].shape[1] / 50.0, frames[0].shape[0] / 50.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(),
                                   animate,
                                   frames=len(frames),
                                   interval=50)
    fn = os.path.join(path, filename)
    anim.save(fn, writer='imagemagick', fps=60)


def render_env(env, path='./', filename='animation_0.gif', steps=1000):
    print("Rendering envrironment...")
    o = env.reset()
    frames = []

    for t in range(steps):
        # render to frames buffer
        frames.append(env.render(mode='rgb_array'))
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

        if done:
            break

    print("Saving environment as GIF...")
    save_frames_as_gif(frames, path, filename)
    print("Saved environment as GIF...")
