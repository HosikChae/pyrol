import matplotlib.pyplot as plt


class DynamicUpdater(object):
    r"""Dynamically update graph for matplotlib plots. Need to have `plt.ion()` to enter
    interactive mode. This is not the most efficient way to render, but it is a good
    workaround when you want to see the plot update as the data streams in.

    API
        close: func, closes the plot and or stops the plot from updating\
        update: func, updates the plot with post-processed data
    """
    plt.ion()

    def close(self, *args, **kwargs):
        """Closes the plot's window or stops the dynamic updates"""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Updates the plot with latest data"""
        raise NotImplementedError

