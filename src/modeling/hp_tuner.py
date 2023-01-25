class HPTuner():

    """
    "Interface" for different hyperparameter tuning techniques.
    
    Attributes:
        - model_fn (function):
            Function that builds and compiles a Neural Network model 
            (tensorflow.keras.Sequential or tensorflow.keras.Model).
            The function should receive a dictionary of hyperparameters and 
            return the compiled model.
        - hp_space (dict):
            Whole hyperparameter space.

        - best_val_loss (float):
            Validation loss of the best-performing hyperparameters.
        - best_hp (dict):
            Best-performing hyperparameters.
        - best_history (dict):
            Training history of the best-performing hyperparameters.

    Methods:
        - tune:
            Tunes hyperparameters.
    """
    
    
    def __init__(self, model_fn, hp_space):
    
        """
        Class constructor: takes as input all class attributes and generates a 
        HPTuner object.
        """
        
        self.model_fn = model_fn
        self.hp_space = hp_space


    def tune(self, train_data, val_data, callbacks, use_gen=False):

        """
        Performs hyperparameter tuning (should be overwritten by classes that 
        implement the interface).

        Parameters:
            - train_data (tuple or DataGenerator):
                Train and validation data.
                If tuple, should be in (samples, labels) format.
            - val_data (tuple or DataGenerator):
                If tuple, should be in (samples, labels) format.
            - callbacks (keras.callbacks list):
                List of callbacks to use during model training.
            - use_gen (bool, default: False):
                Whether or not the provided train and val data are DataGenerators. 
        """

        pass