import numpy as np 

class NormalizationData():

    def __init__(self, data):
        self.data = data

    def soft_max(self):
        exp_data = np.exp(self.data)
        sum_exp_data = np.sum(exp_data)
        return exp_data / sum_exp_data

    def standarization(self):
        mean = np.mean(self.data)
        std_dev = np.std(self.data)
        return (self.data - mean) / std_dev

    def range(self):
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        return (self.data - min_val) / (max_val - min_val)

# # Ejemplo de uso
# data = np.array([1, 2, 3, 4, 5])  # Datos de ejemplo
# nd = NormalizationData(data)

# softmax_result = nd.soft_max()
# standarization_result = nd.standarization()
# range_result = nd.range()

# print("Softmax:", softmax_result)
# print("Estandarización:", standarization_result)
# print("Rango:", range_result)
