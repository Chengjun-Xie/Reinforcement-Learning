import gym
import torch
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image


class environment_management:
    def __init__(self, game, device):
        self.device = device

        # Calling unwrapped gives us access to behind-the-scenes dynamics of the environment
        # that we wouldn’t have access to otherwise.
        self.env = gym.make(game).unwrapped

        # current_screen will track the current screen of the environment at any given time,
        # and when it’s set to None, that indicates that we’re at the start of an episode
        # and have not yet rendered the screen of the initial observation.
        self.current_screen = None

        self.done = False

    def reset(self):
        # initialize all the environment
        self.env.reset()
        self.current_screen = None

    def close(self):
        # close the environment when we’re finished with it
        self.env.close()

    def render(self, mode='human'):
        # return a numpy array version of the rendered screen
        return self.env.render(mode)

    def get_screen_size(self):
        screen = self.get_processed_screen()
        return screen.shape[2:]

    def get_action_size(self):
        return self.env.action_space.n

    def take_action(self, action):
        # we only care about the reward
        # and whether or not the episode ended from taking the given action

        # item() just returns the value of this tensor as a standard Python number,
        # which is what step() expects.
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        # check whether is in the staring state
        return self.current_screen is None

    def get_processed_screen(self):
        # PyTorch expects CHW
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)

        # get cart location
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        cart_location = int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Resize, and add a batch dimension (BCHW)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])

        return resize(screen).unsqueeze(0).to(self.device)

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1


def render_scene():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = 'CartPole-v0'
    em = environment_management(game, device)
    em.reset()
    for i in range(5):
        em.take_action(torch.tensor([1]))
    scene = em.get_state()
    scene = scene.squeeze(0).permute(1, 2, 0).cpu()

    plt.figure()
    plt.imshow(scene, interpolation='none')
    plt.show()


if __name__ == '__main__':
    render_scene()
