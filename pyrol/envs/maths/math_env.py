

class Env(object):

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        # self.action_space.seed(seed)
        # self.observation_space.seed(seed)
        # return [seed]
        pass

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)