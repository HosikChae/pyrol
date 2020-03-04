
class RealEnv(object):

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)