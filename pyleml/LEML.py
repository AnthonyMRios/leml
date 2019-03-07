import abc

class LEML:
    """ Abstract base class for LEML model.
        Subclasses need to override the fit and predict methods.

        Parameters
        ----------
        num_factors : int
            Number of latent "factors" from the labels to be learnt
        num_interations : int
            Number of iterations for the GC to obtain the approximation
        reg_param : float
            Regularization parameter
        stopping_criteria : float
            Stopping parameter
        cg_max_iter : int
            Max iteration for the approximation of GC
        cg_gtol : float
            Parameter for the GC approximation
        verbose : Boolean
            Parameter for printing output during the execution

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, train_data, train_labels):
       """ Train LEML model.

        Parameters
        ----------
        train_data : matrix (n_samples, n_features)
            Matrix representing the data and training instances of it
        train_labels : sparse_matrix (n_samples, n_labels)
            Matrix representing the labels of the training set 
       """

    @abc.abstractmethod
    def predict(self, test_data):
       """ Make predictions using trained LEML mdoel.
       
       Parameters
       ----------
       test_data : matrix (n_samples, n_features)
            Matrix representing the data for testing

        Returns
        -------
        predictions : matrix (n_samples, n_labels)
            Predictions of the labels

       """
    
    @abc.abstractmethod
    def predict_proba(self, test_data):
       """ Make probabilities using trained LEML mdoel.
       
       Parameters
       ----------
       test_data : matrix (n_samples, n_features)
            Matrix representing the data for testing

        Returns
        -------
        predictions : matrix (n_samples, n_labels)
            Probabilities of the labels

       """

    @staticmethod
    def get_instance(backend='single', **extra_args):
        if backend == 'parallel':
            from LEML_parallel import LEMLsf
            return LEMLsf(**extra_args)
        elif backend == 'single':
            from LEML_single import LEMLs
            return LEMLs(**extra_args)

        raise ValueError("Unknown backend: %r (known backends: "
                         "'parallel', 'single')" % backend)
