# This module contains a class that contains metrics for regression validation 

class Validation:
    """
    This class accepts two parameter expected value and predicted value
    It contains methods that performs regression validation for each of the metrics
    -> RME == Mean Square Error
    -> RMSE == Root Mean Square Error
    -> MAE == Mean Absolute Error
    -> RS == Root Squared or Coefficient of Determination
    """

    # the init method
    def __init__(self, *, exp_val: np.ndarray, pred_val: np.ndarray, model_type: str) -> None:
        self.exp_val = exp_val
        self.pred_val = pred_val
        self.model_type = model_type

    def __repr__(self):
        return f"{__class__.__name__} the shape of expected value and predicted val are: \
    \n {self.exp_val.shape}, {self.pred_val.shape}"

    def MSE(self):
        """ The method returns the Mean Square Error"""
        result = np.mean(np.square(np.subtract(self.exp_val, self.pred_val)))
        return result

    def RMSE(self):
        """ The method returns the Root Mean Square Error"""
        result = np.sqrt(np.mean(np.square(np.subtract(self.exp_val, self.pred_val))))
        return result

    def MAE(self):
        """ The method returns the Mean Absolute Error"""
        result = np.mean(np.absolute(self.exp_val - self.pred_val))
        return result

    def RS(self):
        """ This returns the R-squared metric (Coeficient of Determination)"""
        numerator = np.sum(np.square(np.subtract(self.exp_val, self.pred_val)))
        denomerator = np.sum(
            np.square(np.subtract(self.exp_val, np.mean(self.pred_val)))
        )
        result = 1 - (np.divide(numerator, denomerator))
        return result

    def all_metrics(self):
        """ This method returns all the metrics in a dataframe format"""
        dict_ = {
            "Mean Square Error": self.MSE(),
            "Root Mean Square Error": self.RMSE(),
            "Mean Absolute Error": self.MAE(),
            "R-Squared": self.RS(),
        }
        index_ = ["Value"]

        df = (
            pd.DataFrame(dict_, index=index_)
            .T.reset_index()
            .rename(columns={"index": "Metrics", "Value": self.model_type})
        )
        return df
