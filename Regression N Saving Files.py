#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

chosen_degree = 1
year = 2018; last_month = 0
#---------------------------------------------------------------------------------------    

#creating a sum of 'fechamentos no dia' matrix in 'MES'
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
xy_month = dataset.iloc[297:1162, 0:4].values

for i in range (1, len(xy_month[:,3])):
    if xy_month[i, 0] == xy_month[i-1, 0] and xy_month[i, 1] == xy_month[i-1, 1]:
        xy_month[i, 3] = xy_month[i-1, 3] + xy_month[i, 3]
    i = i + 1
del i
#np.savetxt('fechamentos_dia_FULL_acumulado_mes.csv', xy_month, delimiter=',')

# Creating a dictionary for months SUMMED
months_sum = {}
A = np.zeros(shape=(1,4),dtype='int64')

for i in range (1, len(xy_month[:, 0])):
    if xy_month[i, 1] == xy_month[i-1, 1]:
        A = np.append(A, xy_month[i-1, :].reshape(1,-1), axis=0)
    else: 
        months_sum [xy_month[i-1, 0],xy_month[i-1, 1]] = A[1:, :]
        A = np.zeros(shape=(1,4),dtype='int64')
    if i == len(xy_month[:, 0]) - 1:
        A = np.append(A, xy_month[i, :].reshape(1,-1), axis=0)
        months_sum [xy_month[i-1, 0],xy_month[i-1, 1]] = A[1:, :]
    i = i + 1
del A; del i

#---------------------------------------------------------------------------------------
# creating a function to find the best polynomial
def best_polynomial(X, y, max_degree=4, MinMaxScaler=True, weighted=True, 
                    plt_error=False, plt_curve=True, save=False):

    E, W = 10**10, 10**10
    datasize = len(y)
    
    if MinMaxScaler==True: 
        #Applying Feature Scalling
        from sklearn.preprocessing import MinMaxScaler
        norm_X = MinMaxScaler()
        X = norm_X.fit_transform(X)
        norm_y = MinMaxScaler()
        y = norm_y.fit_transform(y)
        #does the linear regression library do the FS by itself ???
        
    #CHOOSING THE BEST ONE
    from sklearn.metrics import mean_absolute_error
    error_list = np.zeros(shape=(1,1))
    for n in range ( 1, max_degree+1 ):
       
        # Fitting Polynomial Regression to the dataset
        from sklearn.preprocessing import PolynomialFeatures
        pol_reg = PolynomialFeatures(degree = n)
        X_pol = pol_reg.fit_transform(X)  #creates polynomial terms
        from sklearn.linear_model import LinearRegression 
        lin_reg = LinearRegression() 
        lin_reg.fit(X_pol, y)  #performs linear regression w/ the pol. terms
        
        #evaluating the model
        e = mean_absolute_error( y_true=y, y_pred=lin_reg.predict(X_pol) )
        error_list = np.append(error_list, e) 
        w = ( (datasize - 1 - n)/(datasize -1) ) * e
        
        if weighted==True:
            if w < W:
                W = w; E = e; N = n
                chosen_lin_reg = lin_reg; chosen_pol_reg = pol_reg
                coef = lin_reg.coef_
        if weighted==False:
            if e < E:
                E = e; W = w; N = n
                chosen_lin_reg = lin_reg; chosen_pol_reg = pol_reg
                coef = lin_reg.coef_
        
        n = n + 1
    
    error_list = error_list.reshape(1,-1)
    error_list = error_list[:,1:]
    
    absolute_error_matrix = y - chosen_lin_reg.predict(X_pol)
    #absolute_error_matrix = np.array(absolute_error_matrix, dtype=float)
    max_errors = np.array([[max(absolute_error_matrix), min(absolute_error_matrix),
                            np.argmax(absolute_error_matrix)/len(y),
                            np.argmin(absolute_error_matrix)/len(y)]])
      
    if plt_error==True: #error visualization
        plt.plot(error_list[0,:], label='variação do erro')
        plt.legend()
        plt.show()
        print(error_list)
    
    if plt_curve==True: #Visualising the Polynomial Regression Results
        X_grid = np.arange(min(X), max(X)+0.1, 0.1).reshape(-1,1)
        plt.scatter(X, y, color = 'red', label='real points')
        plt.plot(X_grid, chosen_lin_reg.predict(chosen_pol_reg.fit_transform(X_grid)), 
                 color = 'blue', label='predicted results') 
        plt.title('Polynomial Regression Results')
        plt.xlabel('Dia %')
        plt.ylabel('Fechamentos %')
        plt.legend()
        plt.show()
        
    if save!=False: #Saving the plot
        X_grid = np.arange(min(X), max(X)+0.1, 0.1).reshape(-1,1)
        plt.scatter(X, y, color = 'red', label='real points')
        plt.plot(X_grid, chosen_lin_reg.predict(chosen_pol_reg.fit_transform(X_grid)), 
                 color = 'blue', label='predicted results') 
        plt.title('Polynomial Regression Results')
        plt.xlabel('Dia %')
        plt.ylabel('Fechamentos %')
        plt.legend()
        plt.savefig('best pol of degree {}, from {}.png'.format(max_degree,save))
        plt.close()
        
    return (N, coef, W, E, error_list, max_errors)

#---------------------------------------------------------------------------------------
# GETTING ALL THE POLYNOMIAL COEFICIENTS

'''dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
# i 297 is the first day of 2016
xy = dataset.iloc[297:1162, 0:4].values'''

Coef_matrix = np.zeros(shape = (1,15)); Dates_matrix = np.zeros(shape = (1,2))
Error_matrix = np.zeros(shape=(1,chosen_degree)); Absolute_error = np.zeros((1,4))

#goes trough years and months, generating the best polinomials
for i in range ( min(xy_month.astype(int)[:, 0]), max(xy_month.astype(int)[:, 0])+1, 1):
    for j in range ( min(xy_month.astype(int)[:, 1]), max(xy_month.astype(int)[:, 1])+1, 1):
        # i = 2016; j = 2
        if i == 2018 and j == 10:
            break
        
        # Substitute i and j for the first month you do not want the regression to be performed
        
        serie = months_sum[i,j]        
        X = serie[:,2].reshape(-1, 1)
        y = serie[:,-1].reshape(-1, 1)
        result = best_polynomial(X,y, 
                            max_degree = chosen_degree, 
                            weighted = True, 
                            MinMaxScaler = True,
                            plt_curve=True,
                            save='{}{}'.format(i,j))
        
        # to different degrees change the variable "chosen_degree" above this loop!
        
        C = result[1]
        e = result[4]
        a = result[5]
        print(i,j)
        f = 15 - len(C[0,:]); F = np.zeros((1,f))
        C = np.append(C, F, axis=1 )
        Coef_matrix = np.append(Coef_matrix, C, axis=0)
        Error_matrix = np.append(Error_matrix, e, axis=0)
        Absolute_error = np.append(Absolute_error, a)
        t=np.array([[i,j]])
        Dates_matrix = np.append(Dates_matrix, t, axis=0)
del i,j,serie,X,y,result,C,e,a,f,F,t

Coef_matrix = Coef_matrix[1:,:]
Dates_matrix = Dates_matrix[1:,:]
Absolute_error = np.array(Absolute_error[4:].reshape(-1,4), dtype=float)
Final_matrix = np.append(Dates_matrix, Coef_matrix, axis=1)
Error_matrix = Error_matrix[1:,:]
del Dates_matrix

# SAVES Final_matrix as CSV file
np.savetxt('Final_matrix, chosen_degree={}.csv'.format(chosen_degree),
           Final_matrix, delimiter=',')
np.savetxt('Error_matrix, chosen_degree={}.csv'.format(chosen_degree),
           Error_matrix, delimiter=',')

#---------------------------------------------------------------------------------------
#PLOTS OF ... EVERYTHING

#BOXPLOT of the coefficients
plt.boxplot(Coef_matrix[:,0:5]); 
plt.title('boxplot of the COEFFICIENTS for pol_degree = chosen_degree')
plt.ylabel("coefficients's values")
plt.xlabel('n coeficients (X^(n-1))')
plt.savefig('boxplot of the COEFFICIENTS for pol_degree = chosen_degree.png')
plt.show()

#BOXPLOT of the mean absolute errors
plt.boxplot(Error_matrix[:,:])
plt.title('boxplot of the mean absolute ERROR(S)')
plt.ylabel('associated error values')
plt.xlabel('polynomial degree')
plt.savefig('boxplot of the mean absolute ERROR(S).png')
plt.show()

#BOXPLOT of the absolute errors
Absolute_error[:,1] = abs(Absolute_error[:,1])
plt.boxplot(Absolute_error[:,0:2])
plt.title('boxplot of the absolute ERROR(S)')
plt.ylabel('associated error values')
plt.xlabel('1=max pos        2=abs min neg')
plt.savefig('boxplot of the absolute ERROR(S).png')
plt.show()

#BOXPLOT of the absolute errors place inside month
Absolute_error[:,1] = abs(Absolute_error[:,1])
plt.boxplot(Absolute_error[:,2:4])
plt.title('boxplot of the absolute ERROR(S)')
plt.ylabel('associated error values')
plt.xlabel('1=max pos        2=abs min neg')
plt.savefig('boxplot of the absolute ERROR(S) place inside month.png')
plt.show()

#AVERAGE COEFICIENTS
avg_coef = np.zeros(shape=(len(Coef_matrix), 5))
for i in range (1, 5):
    avg_coef[:,i] = np.average(Coef_matrix[:,i], axis=0)
del i
np.savetxt('avg_coef, chosen_degree={}.csv'.format(chosen_degree),
           avg_coef[0,:], delimiter=',')

#COEFFICIENTS EVOLUTION    
plt.plot(Coef_matrix[:,1], label='X^1'); plt.plot(Coef_matrix[:,2], label='X^2'); 
plt.plot(Coef_matrix[:,3], label='X^3'); plt.plot(Coef_matrix[:,4], label='X^4')
plt.plot(avg_coef[:,1], label='avg of X^1'); plt.plot(avg_coef[:,2], label='avg of X^2'); 
plt.plot(avg_coef[:,3], label='avg of X^3'); plt.plot(avg_coef[:,4], label='avg of X^4')
plt.title('coeficients variation during analized months')
plt.xlabel('months analized')
plt.ylabel('coeficients values')
plt.legend()
plt.savefig('coeficients variation during analized months.png')
plt.show()

#COEFFICIENTS VALUES
plt.bar(0, avg_coef[0,0], label='avg of X^0'); 
plt.bar(1, avg_coef[0,1], label='avg of X^1'); plt.bar(2, avg_coef[0,2], label='avg of X^2'); 
plt.bar(3, avg_coef[0,3], label='avg of X^3'); plt.bar(4, avg_coef[0,4], label='avg of X^4')
plt.title('coeficients average values during all analized months')
plt.xlabel('months analized')
plt.ylabel('coeficients values')
plt.legend()
plt.savefig('coeficients average values during all analized months.png')
plt.show()

#Comparison of LINEAR REGRESSION vs POLYNOMIAL REGRESSION
Final_curve = np.arange(start=0, stop=1.01, step=(1/30)).reshape(-1,1)
Final_curve = np.append(Final_curve, np.zeros(shape=(31,2)), axis=1)
for i in range (31):
    v = 0
    d = 0
    for j in range (1, chosen_degree+1):
        v = v + avg_coef[0,j]*Final_curve[i,0]**j
        Final_curve[i,1] = v
        d = Final_curve[i,0] - Final_curve[i,1]
        Final_curve[i,2] = d
del i, j, v, d

np.savetxt('Final_curve, chosen_degree={}.csv'.format(chosen_degree),
           Final_curve, delimiter=',')

plt.plot(Final_curve[:,0], np.zeros(shape=(31,1))[:,0], label='zero line')
plt.plot(Final_curve[:,0], Final_curve[:,0], label='linear curve')
plt.plot(Final_curve[:,0], Final_curve[:,1], label='average curve')
plt.plot(Final_curve[:,0], Final_curve[:,2], label='linear - average')
plt.title('Comparison of LINEAR REGRESSION vs POLYNOMIAL REGRESSION')
plt.legend()
plt.savefig('Comparison of LINEAR REGRESSION vs POLYNOMIAL REGRESSION.png')
plt.show()

#-------------------

#AVERAGE COEFICIENTS for determined YEAR
avg_coef_year = np.zeros((1,5)); m = 0; 
for i in range ( len(Final_matrix[:,0]) ):
    if Final_matrix[i,0] == year and Final_matrix[i,1] > last_month:
        avg_coef_year[0,:] = avg_coef_year[0,:] + Final_matrix[i,2:7]
        m = m + 1
avg_coef_year[0,:] = avg_coef_year[0,:] / m
del m, i
np.savetxt('avg_coef_year{}_month_{}, chosen_degree={}.csv'.format(year,
           last_month, chosen_degree),
           avg_coef_year[0,:], delimiter=',')
    
#COEFFICIENTS VALUES for last period
plt.bar(0, avg_coef[0,0], label='avg of X^0'); 
plt.bar(2, avg_coef[0,1], label='avg of X^1'); plt.bar(4, avg_coef[0,2], label='avg of X^2'); 
plt.bar(6, avg_coef[0,3], label='avg of X^3'); plt.bar(8, avg_coef[0,4], label='avg of X^4')
plt.bar(1, avg_coef_year[0,0], label='new avg of X^0'); 
plt.bar(3, avg_coef_year[0,1], label='new avg of X^1'); plt.bar(5, avg_coef_year[0,2], label='new avg of X^2'); 
plt.bar(7, avg_coef_year[0,3], label='new avg of X^3'); plt.bar(9, avg_coef_year[0,4], label='new avg of X^4')
plt.title('coeficients average values during last period')
plt.xlabel('months analized')
plt.ylabel('coeficients values')
plt.legend(loc=3)
plt.savefig('coeficients average values during last period.png')
plt.show()

#Comparison of LINEAR REGRESSION vs POLYNOMIAL REGRESSION WITH AVERAGE FOR ONE YEAR
Final_curve_year = np.arange(start=0, stop=1.01, step=(1/30)).reshape(-1,1)
Final_curve_year = np.append(Final_curve_year, np.zeros(shape=(31,2)), axis=1)
for i in range (31):
    v = 0
    d = 0
    for j in range (1, chosen_degree+1):
        v = v + avg_coef_year[0,j]*Final_curve_year[i,0]**j
        Final_curve_year[i,1] = v
        d = Final_curve_year[i,0] - Final_curve_year[i,1]
        Final_curve_year[i,2] = d
del i, j, v, d

np.savetxt('Final_curve_year{}_month_{}, chosen_degree={}.csv'.format(year,
           last_month, chosen_degree),
           Final_curve_year[0,:], delimiter=',')
del year, last_month

plt.plot(Final_curve_year[:,0], np.zeros(shape=(31,1))[:,0], label='zero line')
plt.plot(Final_curve_year[:,0], Final_curve_year[:,0], label='linear curve')
plt.plot(Final_curve_year[:,0], Final_curve_year[:,1], label='average curve')
plt.plot(Final_curve_year[:,0], Final_curve_year[:,2], label='linear - average')
plt.title('Comparison of LINEAR REGRESSION vs POLYNOMIAL REGRESSION with avg_coef_YEAR')
plt.legend()
plt.savefig('Comparison of LINEAR REGRESSION vs POLYNOMIAL REGRESSION with avg_coef_YEAR.png')
plt.show()

#---------------------------------------------------------------------------------------
# UNUSED BUT MAYBE USEFUL CODES
'''
# IMPORTING THE DATASET
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
# i 297 is the first day of 2016
xy = dataset.iloc[297:, 0:4].values

#creating a sum of 'fechamentos no dia' matrix in 'MES'
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
xy_month = dataset.iloc[297:, 0:4].values
for i in range (1, len(xy_month[:,3])):
    if xy_month[i, 0] == xy_month[i-1, 0] and xy_month[i, 1] == xy_month[i-1, 1]:
        xy_month[i, 3] = xy_month[i-1, 3] + xy_month[i, 3]
    i = i + 1
del i
    
#creating a sum of 'fechamentos no dia' matrix in 'ANO'
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
xy_year = dataset.iloc[297:, 0:4].values
for i in range (1, len(xy_year[:,3])):
    if xy_year[i, 0] == xy_year[i-1, 0]:
        xy_year[i, 3] = xy_year[i-1, 3] + xy_year[i, 3]
    i = i + 1
del i
    
#creating a sum of 'fechamentos no dia' matrix in EVERYTHING 
"""why does the sum_xy change together with the xy_sum ???"""
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
xy_tot = dataset.iloc[297:, 0:4].values
for i in range (1, len(xy_tot[:,3])):
    xy_tot[i, 3] = xy_tot[i-1, 3] + xy_tot[i, 3]
    i = i + 1
del i
'''

#---------------------------------------------------------------------------------------
'''
# SLICING THE TIMESERIES

# Creating a dictionary for years
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
xy = dataset.iloc[297:, 0:4].values
years = {}
A = np.zeros(shape=(1,4),dtype='int64')

for i in range (1, len(xy[:, 0])):
    if xy[i, 0] == xy[i-1, 0]:
        A = np.append(A, xy[i-1, :].reshape(1,-1), axis=0)
    else:
        years [xy[i-1, 0]] = A[1:, :]
        A = np.zeros(shape=(1,4),dtype='int64')
    if i == len(xy[:, 0]) - 1:
        A = np.append(A, xy[i, :].reshape(1,-1), axis=0)
        years [xy[i-1, 0]] = A[1:, :]
    i = i + 1
del A; del i
    
# Creating a dictionary for months
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
xy = dataset.iloc[297:, 0:4].values
months = {}
A = np.zeros(shape=(1,4),dtype='int64')

for i in range (1, len(xy[:, 0])):
    if xy[i, 1] == xy[i-1, 1]:
        A = np.append(A, xy[i-1, :].reshape(1,-1), axis=0)
    else: 
        months [xy[i-1, 0],xy[i-1, 1]] = A[1:, :]
        A = np.zeros(shape=(1,4),dtype='int64')
    if i == len(xy[:, 0]) - 1:
        A = np.append(A, xy[i, :].reshape(1,-1), axis=0)
        months [xy[i-1, 0],xy[i-1, 1]] = A[1:, :]
    i = i + 1
del A; del i
    
# Creating a dictionary for years SUMMED
dataset = pd.read_csv('fechamentos_dia_FULL.csv', sep = "[-,]", engine='python')
xy_year = dataset.iloc[297:, 0:4].values
for i in range (1, len(xy_year[:,3])):
    if xy_year[i, 0] == xy_year[i-1, 0]:
        xy_year[i, 3] = xy_year[i-1, 3] + xy_year[i, 3]
    i = i + 1
del i
years_sum = {}
A = np.zeros(shape=(1,4),dtype='int64')

for i in range (1, len(xy_year[:, 0])):
    if xy_year[i, 0] == xy_year[i-1, 0]:
        A = np.append(A, xy_year[i-1, :].reshape(1,-1), axis=0)
    else:
        years_sum [xy_year[i-1, 0]] = A[1:, :]
        A = np.zeros(shape=(1,4),dtype='int64')
    if i == len(xy_year[:, 0]) - 1:
        A = np.append(A, xy_year[i, :].reshape(1,-1), axis=0)
        years_sum [xy_year[i-1, 0]] = A[1:, :]
    i = i + 1
del A; del i'''

'''
# RESULTS FOR A SERIES:
serie = months_sum[2018,4]
grid = np.arange( len(serie[:,0]))
plt.scatter(grid, serie[:,-1], color='red', label='real points')
plt.plot(grid, serie[:,-1], color='blue', label='linear interpolation')
plt.title('evolução de vendas no mês da "serie"')
plt.legend()
plt.show()
del grid
'''
#---------------------------------------------------------------------------------------