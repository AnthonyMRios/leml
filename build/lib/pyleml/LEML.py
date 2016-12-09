import abc

class LEML:
    """ Abstract base class for LEML model.
        Subclasses need to override the fit and predict methods.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, train_data, train_labels):
       """ Train LEML model. """

    @abc.abstractmethod
    def predict(self, test_data):
       """ Make predictions using trained LEML mdoel. """

    @staticmethod
    def get_instance(backend='single', **extra_args):
        if backend == 'parallel':
            from LEML_parallel import LEMLp
            return LEMLp(**extra_args)
        elif backend == 'single':
            from LEML_single import LEMLs
            return LEMLs(**extra_args)

        raise ValueError("Unknown backend: %r (known backends: "
                         "'parallel', 'single')" % backend)
