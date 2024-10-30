import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress, t
import matplotlib.pyplot as plt

class PolyFit:
    def __init__(self):
        pass

    def fit(self, x, y, degree):
        """
        Fit a polynomial of a specified degree to the rate data.
        
        Parameters:
        - x (array-like): Time data points.
        - y (array-like): Concentration data points corresponding to the time data.
        - degree (int): The degree of the polynomial to fit.
        
        Returns:
        - coefficients (array): Coefficients of the fitted polynomial, starting with the highest degree.
        """
        self.coefficients = np.polyfit(x, y, degree)
        self.degree = degree

        # Calculate residuals
        self.y_fit = np.polyval(self.coefficients, x)
        residuals = y - self.y_fit
        
        # Variance estimate
        self.residual_var = np.var(residuals, ddof=len(self.coefficients))
        
        # Store data for confidence interval calculations
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        
        return self.coefficients
    
    def gradient(self, x, y):
        

    def predict(self, x):
        """
        Predict the concentration based on the fitted polynomial.
        
        Parameters:
        - x (array-like): Time data points to predict the concentration for.
        
        Returns:
        - y (array): Predicted concentration values.
        """
        y = np.polyval(self.coefficients, x)
        return y

    def confidence_intervals(self, x, confidence=0.95):
        """
        Calculate confidence intervals for predictions.
        
        Parameters:
        - x (array-like): Time data points for prediction.
        - confidence (float): Confidence level for the interval.
        
        Returns:
        - lower_bound (array): Lower bound of the confidence interval.
        - upper_bound (array): Upper bound of the confidence interval.
        """
        mean_x = np.mean(self.x)
        n = len(self.x)
        dof = max(0, n - self.degree - 1)  # degrees of freedom
        t_val = t.ppf((1 + confidence) / 2., dof)
        
        # Predicted values
        y_pred = self.predict(x)

        # Standard error of predictions
        se_pred = np.sqrt(self.residual_var * 
                          (1/n + (x - mean_x)**2 / np.sum((self.x - mean_x)**2)))
        
        # Calculate confidence intervals
        lower_bound = y_pred - t_val * se_pred
        upper_bound = y_pred + t_val * se_pred
        
        return lower_bound, upper_bound

    def plot(self, x, y, adjust_intervals=False):
        """
        Plot the concentration data, fitted polynomial, and confidence intervals.
        
        Parameters:
        - x (array-like): Time data points.
        - y (array-like): Concentration data points corresponding to the time data.
        - adjust_intervals (bool): If True, artificially widens confidence intervals for visibility.
        """
        plt.scatter(x, y, color='blue', label='Data')
        plt.plot(x, self.predict(x), color='red', label='Fitted Polynomial')

        # Calculate and plot confidence intervals
        lower_bound, upper_bound = self.confidence_intervals(x)
        if adjust_intervals:
            lower_bound -= 0.1
            upper_bound += 0.1
        plt.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.5, label='95% Confidence Interval')
        
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.legend()
        plt.show()
    
class IntegralMethod:
    def __init__(self, time, concentration):
        """
        Initialize with time and concentration data.
        
        Parameters:
        - time (array-like): Time data points.
        - concentration (array-like): Concentration data points.
        """
        self.time = np.array(time)
        self.concentration = np.array(concentration)
    
    def zero_order(self):
        """
        Calculate the rate constant assuming a zero-order reaction.
        
        Returns:
        - rate_constant (float): The rate constant for a zero-order reaction.
        - slope (float): Slope of the best-fit line for [A] vs. time.
        - intercept (float): Intercept of the best-fit line.
        """
        # [A] vs. time
        slope, intercept, r_value, _, _ = linregress(self.time, self.concentration)
        
        rate_constant = -slope
        return rate_constant, intercept
        
    def first_order(self):
        """
        Calculate the rate constant assuming a first-order reaction.
        
        Returns:
        - rate_constant (float): The rate constant for a first-order reaction.
        - slope (float): Slope of the best-fit line for ln([A]_0 / [A]) vs. time.
        - intercept (float): Intercept of the best-fit line.
        """
        # ln([A]_0 / [A]) vs. time
        ln_concentration_ratio = np.log(self.concentration[0] / self.concentration)
        slope, intercept, r_value, _, _ = linregress(self.time, ln_concentration_ratio)
        
        rate_constant = slope  # for first-order reactions, k = slope
        return rate_constant, intercept

    def second_order(self):
        """
        Calculate the rate constant assuming a second-order reaction.
        
        Returns:
        - rate_constant (float): The rate constant for a second-order reaction.
        - slope (float): Slope of the best-fit line for (1/[A]) - (1/[A]_0) vs. time.
        - intercept (float): Intercept of the best-fit line.
        """
        # (1/[A]) - (1/[A]_0) vs. time
        inverse_concentration_diff = (1 / self.concentration) - (1 / self.concentration[0])
        slope, intercept, r_value, _, _ = linregress(self.time, inverse_concentration_diff)
        
        rate_constant = slope  # for second-order reactions, k = slope
        return rate_constant, intercept

    def plot(self, order=1):
        """
        Plot the data for the integral method with best-fit line to determine rate constant.
        
        Parameters:
        - order (int): Order of the reaction (1 or 2).
        """
        if order == 1:
            # First-order reaction plot: ln([A]_0 / [A]) vs. time
            ln_concentration_ratio = np.log(self.concentration[0] / self.concentration)
            plt.plot(self.time, ln_concentration_ratio, 'o', label='Data (1st Order)')
            
            rate_constant, slope, intercept = self.first_order_rate_constant()
            plt.plot(self.time, slope * self.time + intercept, 'r--', label=f'Fit: k = {rate_constant:.4f}')
            plt.ylabel("ln([A]_0 / [A])")
        
        elif order == 2:
            # Second-order reaction plot: (1/[A]) - (1/[A]_0) vs. time
            inverse_concentration_diff = (1 / self.concentration) - (1 / self.concentration[0])
            plt.plot(self.time, inverse_concentration_diff, 'o', label='Data (2nd Order)')
            
            rate_constant, slope, intercept = self.second_order_rate_constant()
            plt.plot(self.time, slope * self.time + intercept, 'r--', label=f'Fit: k = {rate_constant:.4f}')
            plt.ylabel("(1/[A]) - (1/[A]_0)")
        
        plt.xlabel("Time")
        plt.legend()
        plt.show()

class NonLinFit():
    '''
    Class for nonlinear regression analysis of rate data.
    a model_func for rate expression is needed to perform the regression.
    '''
    def __init__(self):
        pass

    def polynomial_fit(self, x, y, degree):
        """
        Fit a polynomial of a specified degree to the rate data.
        
        Parameters:
        - x (array-like): Time data points.
        - y (array-like): Concentration data points corresponding to the time data.
        - degree (int): The degree of the polynomial to fit.
        
        Returns:
        - coefficients (array): Coefficients of the fitted polynomial, starting with the highest degree.
        """
        self.coefficients = np.polyfit(x, y, degree)
        return self.coefficients
    
    
    def gradient(self):
        """
        Calculate the gradient (derivative) of the fitted polynomial.
        
        Returns:
        - gradient_coeffs (array): Coefficients of the gradient polynomial, starting with the highest degree.
        """
        if hasattr(self, 'coefficients'):
            self.gradient_coeffs = np.polyder(self.coefficients)
            return self.gradient_coeffs
        else:
            raise AttributeError("Polynomial must be fitted before calculating gradient.")

    def calc_rate(self):
        """
        Calculate the rate of the reaction using the gradient polynomial.

        dCdt = p_1 + 2*p_2*t + 3*p_3*t^2 + ...
        
        Returns:
        - rate (array): Rate of the reaction at each time point.
        """
        if hasattr(self, 'gradient_coeffs'):
            self.rate = np.polyval(self.gradient_coeffs, self.x)
            return self.rate
        else:
            raise AttributeError("Gradient must be calculated before calculating rate.")
     

    def fit(self, x, y, model_func, initial_guess):
        """
        Perform nonlinear regression on the rate data given a model function.
        
        Parameters:
        - x (array-like): Time data points.
        - y (array-like): Concentration data points corresponding to the time data.
        - model_func (callable): The model function, such as for fractional order or exponential kinetics.
        - initial_guess (array-like): Initial guess for the parameters in the model function.
        
        Returns:
        - params (array): Optimal values for the parameters in model_func.
        - covariance (2D array): Covariance of params.
        """
        if hasattr(self, 'rate'):
            self.params, self.covariance = curve_fit(model_func, x, self.rate, p0=initial_guess)
            return self.params, self.covariance
        else:
            raise AttributeError("Rate must be calculated before performing nonlinear regression.")

    def predict(self, x, model_func):
        """
        Predict the concentration based on the fitted model.
        
        Parameters:
        - x (array-like): Time data points to predict the concentration for.
        - model_func (callable): The model function used in nonlinear regression.
        
        Returns:
        - y (array): Predicted rate values.
        """
        rate = model_func(x, *self.params)
        return rate

    def plot(self, x, y, model_func):
        """
        Plot the concentration data and the fitted model.
        
        Parameters:
        - x (array-like): Concentration data points.
        - y (array-like): rate data points corresponding to the time data.
        - model_func (callable): The model function used in nonlinear regression.
        """
        plt.scatter(x, y, color='blue', label='Data')
        
        fitted_concentration = model_func(x, *self.params)
        plt.plot(x, fitted_concentration, color='red', label='Fitted Model')
        
        plt.xlabel("Concentration")
        plt.ylabel("Rate")
        plt.legend()
        plt.show()

class InitialRate():
    def __init__(self):
        pass

    def fit(self, x, y):
        """
        Fit a linear model to the initial rate data.
        
        Parameters:
        - x (array-like): Concentration data points.
        - y (array-like): Initial rate data points corresponding to the concentration data.
        
        Returns:
        - rate_constant (float): The rate constant for the reaction.
        - intercept (float): Intercept of the best-fit line.
        """
        slope, intercept, r_value, _, _ = linregress(x, y)
        rate_constant = slope
        return rate_constant, intercept

    def predict(self, x):
        """
        Predict the initial rate based on the fitted linear model.
        
        Parameters:
        - x (array-like): Concentration data points to predict the initial rate for.
        
        Returns:
        - y (array): Predicted initial rate values.
        """
        y = self.rate_constant * x + self.intercept
        return y

    def plot(self, x, y):
        """
        Plot the initial rate data and the fitted linear model.
        
        Parameters:
        - x (array-like): Concentration data points.
        - y (array-like): Initial rate data points corresponding to the concentration data.
        """
        plt.scatter(x, y, color='blue', label='Data')
        plt.plot(x, self.predict(x), color='red', label='Fitted Model')
        
        plt.xlabel("Concentration")
        plt.ylabel("Initial Rate")
        plt.legend()
        plt.show()