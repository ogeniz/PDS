import numpy as np

def conv_ofc(a,b):
    la = np.int16(len(a)) # dimensão do vetor de entrada
    lb = np.int16(len(b)) # dimensão vetor de resposta ao \
      # impulso    
    ly = (la + lb) - 1 # dimensão do vetor da resposta do sistema
    y = np.zeros(ly,dtype = np.int8) # preenche o vetor com zeros

    for n in np.arange(0, la) : # n vairia no intervalo [0,lx]
        for k in np.arange(0, lb) : # k vairia no intervalo [0,lh]
            y[n + k] += a[n] * b[k] # calcula a convolução \
              # entre x[n] e  h[n]
    return y


def diff_plot(a,b):
    import matplotlib.pyplot as plt

    y = conv_ofc(b,a)
    la,lb,ly = len(a),len(b),len(y)
    
    ax1 = plt.subplot(3,1,1)
    ax1.set_xlim([-1,len(y) + 1])
    ax1.set_ylim([y.min() - 1,y.max() + 1])
    ax1.grid('on')
    plt.stem(np.arange(0,ly),y,
                 markerfmt = 'ro',linefmt = 'g--',
                 basefmt = 'm:' )
    plt.title('Discrete Convolution')
    plt.ylabel('y[n]')
    
    ax2 = plt.subplot(3,1,2)
    ax2.set_xlim([-1,len(b) + 1])
    ax2.set_ylim([b.min() - 1,b.max() + 1])
    ax2.grid('on')
    plt.stem(np.arange(0,lb),b,
                 markerfmt = 'ro',linefmt = 'g--',
                 basefmt = 'm:' )
    plt.ylabel('h[n]')

    ax3 = plt.subplot(3,1,3)
    ax3.set_xlim([-1,len(a) + 1])
    ax3.set_ylim([a.min() - 1,a.max() + 1])
    ax3.grid('on')
    plt.stem(np.arange(0,la),a,
                 markerfmt = 'ro',linefmt = 'g--',
                 basefmt = 'm:' )
    plt.ylabel('x[n]')
    plt.xlabel('n')

    plt.show()

def f_recur(n,y):
    i = n - 1
    if i :
        f_recur(i,y)
        y[i] = 2.0*(1 if not (i - 1) else 0) + (1 if not (i - 3) else 0) + \
          0.5*y[(i - 1) if not (i - 1) else i] - \
          0.25*y[(i - 2) if not (i - 2) else i] 
    else:
        y[0] = 1
    return y

def main():
    #    n = np.arange(0,25)
    #    x = np.array(5 + 3*np.cos(0.2*np.pi*n) + \
    #    4*np.sin(0.6*np.pi*n),dtype = float)
    #   diff_plot(x,h)
    a = np.int8(4)
    z = np.zeros(a)    
    print(f_recur(a,z))
    
if __name__ == "__main__":
    main()
