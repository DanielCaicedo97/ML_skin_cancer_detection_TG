import statsmodels.stats.diagnostic as smd

class GaussianTest():
    
    def __init__(self, data):
        self.data = data
    
    def lilliefors(self):
        p_value = smd.lilliefors(self.data)[1]
        return p_value > 0.05
    
    def Kolmogorov_Smirnov(self):
        p_value = smd.kstest_normal(self.data)[1]
        return p_value > 0.05
    
    def anderson_darling(self):
        statistic = smd.anderson_statistic(self.data, dist='norm')
        return statistic <= 0.5  # Si el estadístico es mayor que 0.5, entonces los datos no son normales

# # Ejemplo de uso
# data = np.random.normal(loc=0, scale=1, size=1000)  # Datos de ejemplo
# gt = GaussianTest(data)

# is_gaussian_lilieffors = gt.lilliefors()
# is_gaussian_kolmorov = gt.Kolmogorov_Smirnov()
# is_gaussian_anderson = gt.anderson_darling()

# print("¿Los datos son gaussianos según el test de Liliefors?", not is_gaussian_lilieffors)
# print("¿Los datos son gaussianos según el test de Kolmogorov-Smirnov?", not is_gaussian_kolmorov)
# print("¿Los datos son gaussianos según el test de Anderson-Darling?", not is_gaussian_anderson)
