from matplotlib import pyplot as plt


def show_frame(env, state):
    """show a frame in the given environment"""
    frame = env.render(mode='rgb_array')
    plt.axis('off')
    plt.title(f"State: {state}")
    plt.imshow(frame)
    plt.show()
