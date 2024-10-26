# Basics
import random
from tqdm import tqdm
from tensorflow.keras.backend import clear_session

# Custom
from hp_tuner import HPTuner


class RandomTuner(HPTuner):

    """
    Subclass of HPTuner (implements the "interface") which performs hyperparameter 
    tuning through Random Search.
    
    Additional attributes:
        - n_models (int):
            Number of models to generate and evaluate.

    Overwritten methods:
        - tune:
            Implements Random Search.
    """

    def __init__(self, model_fn, hp_space, n_epochs, n_models):

        """
        Class constructor: takes as input all class attributes and generates a 
        RandomTuner object.
        """

        super().__init__(model_fn, hp_space, n_epochs)
        self.n_models = n_models

    
    def tune(self, train_data, val_data, callbacks, metrics=['accuracy'], use_gen=False):

        """
        Performs hyperparameter tuning with a Random Search: a fixed number of 
        random hyperparameters is selected from the hyperparameter space and tested.
        
        Only the best-performing configuration of hyperparameters is considered 
        as result.
        
        The performance of the hyperparameters is given by the validation loss 
        (the lower the better).
        
        Returns:
            - best_hp (dict):
                Best-performing hyperparameters.
            - best_history():
                Training history for the best-performing hyperparameters.
        """
    
        res = []

        for _ in tqdm(range(self.n_models), desc='Random Search: '):

            clear_session()

            random_hp = {k: random.choice(self.hp_space[k]) for k in self.hp_space.keys()}
            model = self.model_fn(
                hp=random_hp,
                input_len=self.trace_len,
                n_classes=self.n_classes,
                metrics=metrics
            )

            if use_gen:
                history = model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=self.n_epochs,
                    batch_size=random_hp['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                ).history
            else:
                history = model.fit(
                    train_data[0], # x_train
                    train_data[1], # y_train
                    validation_data=(val_data[0], val_data[1]), # (x_val, y_val)
                    epochs=self.n_epochs,
                    batch_size=random_hp['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                ).history
            
            # TODO: if ModelCheckpoint in callbacks, load the saved model and 
            # use it to evaluate y_val (and gain val_loss for ranking)

            # If ModelCheckpoint is not in callbacks
            val_loss = history['val_loss'][-1]                 
            res.append((val_loss, random_hp, history))
            
        res.sort(key=lambda x: x[0])
        _, best_hp, best_history = res[0]
        
        return best_hp, best_history