import numpy as np


def smooth_pulse(t,amp,t_start=0.5,duration=2.0):
    """Compactly supported C^2 sin^4 pulse."""
    if t_start<=t<=t_start+duration:
        phase=np.pi*(t-t_start)/duration
        return np.array([amp*np.sin(phase)**4])

    return np.zeros(1)


def smooth_pulse_average(t_0,t_1,amp,t_start=0.5,duration=2.0):
    """Exact average of smooth_pulse over [t_0,t_1]."""
    if t_1<=t_0:
        raise ValueError("Require t_1>t_0.")

    a=max(t_0,t_start)
    b=min(t_1,t_start+duration)

    if b<=a:
        return np.zeros(1)

    theta_a=np.pi*(a-t_start)/duration
    theta_b=np.pi*(b-t_start)/duration

    def primitive(theta):
        return 3.0*theta/8.0-np.sin(2.0*theta)/4.0+np.sin(4.0*theta)/32.0

    integral=amp*duration/np.pi*(primitive(theta_b)-primitive(theta_a))
    return np.array([integral/(t_1-t_0)])