import numpy as np
def spline_fit(xp,yp):

        def akima(x, y):
            n = len(x)
            var = np.zeros((n + 3))
            z = np.zeros((n))
            co = np.zeros((n, 4))
            for i in range(n - 1):
                var[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            var[n + 1] = 2 * var[n] - var[n - 1]
            var[n + 2] = 2 * var[n + 1] - var[n]
            var[1] = 2 * var[2] - var[3]
            var[0] = 2 * var[1] - var[2]

            for i in range(n):
                wi1 = abs(var[i + 3] - var[i + 2])
                wi = abs(var[i + 1] - var[i])
                if (wi1 + wi) == 0:
                    z[i] = (var[i + 2] + var[i + 1]) / 2
                else:
                    z[i] = (wi1 * var[i + 1] + wi * var[i + 2]) / (wi1 + wi)

            for i in range(n - 1):
                dx = x[i + 1] - x[i]
                a = (z[i + 1] - z[i]) * dx
                b = y[i + 1] - y[i] - z[i] * dx
                co[i, 0] = y[i]
                co[i, 1] = z[i]
                co[i, 2] = (3 * var[i + 2] - 2 * z[i] - z[i + 1]) / dx
                co[i, 3] = (z[i] + z[i + 1] - 2 * var[i + 2]) / dx ** 2
            co[n - 1, 0] = y[n - 1]
            co[n - 1, 1] = z[n - 1]
            co[n - 1, 2] = 0
            co[n - 1, 3] = 0
            return co

        p_lst = [lambda x_, c=c, x0=x0: np.poly1d(c[::-1])(x_-x0) for c,x0 in zip(akima(xp,yp), xp)]
        
        def spline(x):
            y = np.empty_like(x)+np.nan
            segment = np.searchsorted(xp,x, 'right')-1
            for i in np.unique(segment):
                m = segment==i
                y[m] = p_lst[i](x[m])
            return y
#         def coef2spline(x, xp, co):
#             
#             print (np.searchsorted(xp,x)-1)
#             x, y = [], []
#             for i, c in enumerate(co.tolist()[:-1]):
#                 p = np.poly1d(c[::-1])
#                 z = np.linspace(0, s[i + 1] - s[i ], 10, endpoint=i >= co.shape[0] - 2)
#                 x.extend(s[i] + z)
#                 y.extend(p(z))
#             return y
#         
        return spline
        #x, y, z = [coef2spline(curve_z_nd, akima(curve_z_nd, self.c2def[:, i])) for i in range(3)]
        #return x, y, z
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    x = np.random.randint(0,100,10)
    t = np.arange(0,100,10)
    plt.plot(t,x,'.',label='points')
    
    t_ = np.arange(100)
    spline = spline_fit(t,x)
    print (np.abs(np.diff(np.diff(np.interp(t_, t,x)))).max())
    print (np.abs(np.diff(np.diff(spline(t_)))).max())
    plt.plot(t_, np.interp(t_, t,x))
    plt.plot(t_, spline(t_),label='spline')
    plt.show()
