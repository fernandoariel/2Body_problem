#librerias utilizadas
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

#####################################

#Funciones
def rates(y, t, m1, m2):
    G=6.67259e-20

    R1=np.array([y[0], y[1], y[2]])
    R2=np.array([y[3], y[4], y[5]])

    V1=np.array([y[6], y[7], y[8]])
    V2=np.array([y[9], y[10], y[11]])
   
    r_norm=np.linalg.norm(R2-R1)
    a_1=(G*m2*(R2-R1))/r_norm**3
    a_2=(G*m1*(R1-R2))/r_norm**3
    return V1[0],V1[1], V1[2], V2[0],V2[1],V2[2], a_1[0],a_1[1],a_1[2], a_2[0],a_2[1],a_2[2]


# Función para graficar los resultados
def plot_2body(X_1, Y_1, Z_1, X_2, Y_2, Z_2, X_G, Y_G, Z_G):
    
    #Figura 1
    plt.figure(1)
    ax = plt.axes(projection='3d')
    
    ax.plot3D(X_1, Y_1 ,Z_1, 'red', label='trayectoria de m1')
    ax.plot3D(X_2, Y_2, Z_2, 'green', label='trayectoria de m2')
    ax.plot3D(X_G, Y_G, Z_G, 'blue', label='trayectoria de G')
    plt.title('Movimiento relativo al sistema inercial')


    ax.scatter3D(X_1[0], Y_1[0], Z_1[0], color='red')
    ax.scatter3D(X_1[-1], Y_1[-1], Z_1[-1], color='red')

    ax.scatter3D(X_2[0], Y_2[0], Z_2[0], color='green')
    ax.scatter3D(X_2[-1], Y_2[-1], Z_2[-1], color='green')

    ax.set_xlabel('eje-x (km)')
    ax.set_ylabel('eje-y (km)')
    ax.set_zlabel('eje-z(km)')
       
    plt.legend(loc='lower left')
    plt.savefig('Mov_1.png')
      
    # Figura 2
    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.plot3D(X_2 - X_1, Y_2 - Y_1, Z_2 - Z_1, 'green', label=r'$m_2$')
    ax.plot3D(X_G - X_1, Y_G - Y_1, Z_G - Z_1, 'blue', label= 'G')
    plt.title('Movimiento de m2 y G relativo a m1')


    ax.set_xlabel('eje-x (km)')
    ax.set_ylabel('eje-y (km)')
    ax.set_zlabel('eje-z(km)')
    
        
    plt.legend(loc='upper left')
    plt.savefig('Mov_2.png')

    # Figura 3
    plt.figure(3) 
    ax = plt.axes(projection='3d')
    ax.plot3D(X_1 - X_G, Y_1 - Y_G, Z_1 - Z_G, 'red', label='$m_1$')
    ax.plot3D(X_2 - X_G, Y_2 - Y_G, Z_2 - Z_G, 'green', label= '$m_2$')
    plt.title('Movimiento de m1 y m2 relativo a G')

       
    ax.set_xlabel('eje-x(km)')
    ax.set_ylabel('eje-y (km)')
    ax.set_zlabel('eje-z (km)')
        
    plt.legend(loc='upper right')
    
    plt.show() 
    plt.savefig('Mov_3.png')


#########################################

#datos de entrada
m_1, m_2 = 1e26, 1e26  #Masas m1 y m2

t0, tf = 0, 480     #tiempo inicial y final

R1_0=np.array([0, 0, 0])      #vector R1 inicial  
R2_0=np.array([3000, 0, 0]) #vector R2 inicial

V1_0=np.array([10, 20, 30]) #vector V1 inicial
V2_0=np.array([0, 40, 0])  #vector V2 inicial

 

y_0=np.concatenate([R1_0, R2_0, V1_0 ,V2_0])

#solver

t=np.arange(t0,tf)

x=odeint(rates, y_0, t , args= (m_1, m_2))  

#salida
X_1, Y_1, Z_1=x[:,0], x[:,1], x[:,2]
X_2, Y_2, Z_2=x[:,3], x[:,4], x[:,5]

X_G, Y_G, Z_G=[], [], []
for i in range(len(t)):
    X_G.append((m_1*X_1[i] + m_2*X_2[i])/(m_1+m_2) )
    Y_G.append((m_1*Y_1[i] + m_2*Y_2[i])/(m_1+m_2) )
    Z_G.append((m_1*Z_1[i] + m_2*Z_2[i])/(m_1+m_2) )


#plot
plot_2body(X_1, Y_1, Z_1, X_2, Y_2, Z_2, X_G, Y_G, Z_G)



