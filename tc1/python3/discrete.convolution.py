import numpy as np
import numpy.matlib
import scipy as scipy
import scipy.linalg

def conv_ofc(a,b):
    la = np.int8(len(a)) # dimensão do vetor de entrada
    lb = np.int8(len(b)) # dimensão vetor de resposta ao impulso    
    ly = la + lb - 1 # dimensão do vetor da resposta do sistema
    y = np.zeros(ly,dtype = np.int8) # preenche o vetor com zeros

    for n in np.arange(0, la) : # n vairia no intervalo [0,lx]
        for k in np.arange(0, lb) : # k vairia no intervalo [0,lh]
            y[n + k] += a[n] * b[k] # calcula a convolução entre x[n] e  h[n]
    return y

def conv_tp(a,b):
    la = np.int8(len(a)) # dimensão vetor de resposta ao impulso
    lb = np.int8(len(b)) # dimensão do vetor de entrada    
    ly = la + lb - 1 # dimensão do vetor da resposta do sistema
    H = np.matlib.zeros((ly,lb),dtype = np.float) # cria uma matriz vazia de zeros

    for i in np.arange(0,ly) : # i varia pelas linhas da matriz H
        for j in np.arange(0,lb) : # j varia pelas colunas da matriz H
            if (i >= j and (la > (i - j))) : # se i >= j e lh > (i - j), não execede a dim de h[n]
                    H[i,j] = a[i - j] # preenche a matrix H com valores vindos de h[n]

    y = np.dot(H,b).reshape(ly,1)
    return y,H

def conv_plot(a,b):
    import matplotlib.pyplot as plt

    y = conv_ofc(b,a)
    la,lb,ly = len(a),len(b),len(y)
    
    ax1 = plt.subplot(3,1,1)
    ax1.set_xlim([-1,len(y) + 1])
    ax1.set_ylim([0,y.max() + 10])
    ax1.grid('on')
    plt.stem(np.arange(0,ly),y,
                           markerfmt = 'ro',linefmt = 'g--', basefmt = 'm:' )
    plt.title('Discrete Convolution')
    plt.ylabel('y[n]')
    
    ax2 = plt.subplot(3,1,2)
    ax2.set_xlim([-1,len(b) + 1])
    ax2.set_ylim([0,b.max() + 1])
    ax2.grid('on')
    plt.stem(np.arange(0,lb),b,
                           markerfmt = 'ro',linefmt = 'g--', basefmt = 'm:' )
    plt.ylabel('h[n]')

    ax3 = plt.subplot(3,1,3)
    ax3.set_xlim([-1,len(a) + 1])
    ax3.set_ylim([0,a.max() + 1])
    ax3.grid('on')
    plt.stem(np.arange(0,la),a,
                           markerfmt = 'ro',linefmt = 'g--', basefmt = 'm:' )
    plt.ylabel('x[n]')
    plt.xlabel('n')

    plt.show()
    
def main():
    x = np.array([1,2,3,4,5]) # Vetor de entrada
    # x = np.random.randint(1,9,size=10) # vetor de entrada aleatório
    h = np.array([6,7,8,9]) # Vetor da resposta ao impulso

    y0 = conv_ofc(h,x)
    y1 = np.convolve(x,h) # cálculo da convolução entre x[n] e h[n] pelo scipy
    y2,H = conv_tp(h,x)

    print('My convolution algorithm result',y0,sep = '\n',end = '\n')
    print('Scipy convolution result',y1,sep = '\n',end = '\n')
    print('y = Hx',y2,sep = '\n',end = '\n')
    print('My Toeplitz Matrix',H,sep = '\n',end = '\n')

    conv_plot(x,h)

if __name__ == '__main__':
    main()    
