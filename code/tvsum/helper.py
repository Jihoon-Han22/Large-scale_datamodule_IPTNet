
from tensorboardX import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        """ Extended SummaryWriter Class from tensorboard-pytorch (tensorboardX)
        https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py
        Internally calls self.file_writer

        :param str logdir: Save directory location.
        """
        super(TensorboardWriter, self).__init__(logdir)
        self.logdir = self.file_writer.get_logdir()

    def update_parameters(self, module, step_i):
        """ Add module's parameters' histogram to summary.

        :param torch.nn.Module module: Module from which the parameters will be taken.
        :param int step_i: Step value to record.
        """
        for name, param in module.named_parameters():
            self.add_histogram(name, param.clone().cpu().data.numpy(), step_i)

    def update_loss(self, loss, step_i, name='loss'):
        """ Add scalar data to summary.

        :param float loss: Value to save.
        :param int step_i: Step value to record.
        :param str name: Data identifier.
        """
        self.add_scalar(name, loss, step_i)

    def update_histogram(self, values, step_i, name='hist'):
        """ Add histogram to summary.

        :param torch.Tensor | numpy.ndarray values: Values to build histogram.
        :param int step_i: Step value to record.
        :param str name: Data identifier.
        """
        self.add_histogram(name, values, step_i)

class Average():
    def __init__(self, *keys):
        self.totals = {key: 0.0 for key in keys}
        self.counts = {key: 0 for key in keys}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            assert key in self.totals and key in self.counts
            self.totals[key] += value
            self.counts[key] += 1

    def __getattr__(self, attr):
        assert attr in self.totals and attr in self.counts
        total = self.totals[attr]
        count = self.counts[attr]
        return total / count if count else 0.0




def update_args(args):
    """
    :param args:
    :return:
    """
    return args
