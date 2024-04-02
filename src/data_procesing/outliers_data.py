import numpy as np

class Outliers():

    def __init__(self, data):
        self.data = data

    def remove_outliers_std(self, threshold=3):
        mean = np.mean(self.data)
        std_dev = np.std(self.data)
        lower_bound = mean - threshold * std_dev
        upper_bound = mean + threshold * std_dev
        filtered_data = self.data[(self.data >= lower_bound) & (self.data <= upper_bound)]
        return np.array(filtered_data)

    def remove_outliers_iqr(self):
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_data = self.data[(self.data >= lower_bound) & (self.data <= upper_bound)]
        return filtered_data

# # Ejemplo de uso
# data = np.array([1, 2, 3, 4, 5, 100])  # Datos de ejemplo
# ol = Outliers(data)

# filtered_data_std = ol.remove_outliers_std()
# filtered_data_iqr = ol.remove_outliers_iqr()

# print("Datos filtrados con desviación estándar:", filtered_data_std)
# print("Datos filtrados con IQR:", filtered_data_iqr)
