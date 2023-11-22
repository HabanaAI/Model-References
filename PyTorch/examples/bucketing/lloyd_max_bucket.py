# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

from scipy import integrate

# computes MSE between 2 adjacent decision thresholds (on one segment)
def interval_MSE(x,t1,t2, f):
    return integrate.quad(lambda t: ((t - x)**2) * f(t), t1, t2)[0]

# computes mean squared error on R
def MSE(t,x,f,minval,maxval):
    s = interval_MSE(x[0], minval, t[0], f) + interval_MSE(x[-1], t[-1], maxval, f)
    for i in range(1,len(x)-1):
        s = s + interval_MSE(x[i], t[i-1], t[i], f)
    return s

# t1 and t2 are the boundaries of the interval on which the centroid is calculated
def centroid(t1,t2, f):
    if integrate.quad(f, t1, t2)[0] == 0 or t1 == t2:
        return 0
    else:
        return integrate.quad(lambda t:t*f(t), t1, t2)[0] / integrate.quad(f, t1, t2)[0]

# t is an array containing the initial decision thresholds
# x is an array containing the representation levels
# error_threshold is the threshold to reach for the algorithm to stop
def lloydmax(t,x,error_threshold, pdf,minval,maxval,max_steps=20):
    e = MSE(t,x, pdf,minval,maxval)
    error = [e]
    c = 0
    while e > error_threshold and c < max_steps:
        c = c+1
        if c%2 == 1:
            # adjust thresholds
            for i in range(len(t)):
                t[i] = 0.5 * ( x[i] + x[i+1] )
        else:
            # adjust levels
            x[0] = centroid(minval, t[0], pdf)
            x[-1] = centroid(t[-1],maxval, pdf)
            for i in range(1,len(x)-1):
                x[i] = centroid(t[i-1], t[i],pdf)
        e = MSE(t,x, pdf,minval,maxval)
        error.append(e)
    return x,t,error


