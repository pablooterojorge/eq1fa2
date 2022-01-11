def ret_vs_vrms(ii0index, ii0, index2vrms = 20):
    vrms = ii0index / index2vrms
    maximums = signal.argrelextrema(ii0, np.greater)[0]
    minimums = signal.argrelextrema(ii0, np.less)[0]
    ret = np.zeros(len(ii0))
    for i in range(len(ii0)):
        if vrms[i] <=  vrms[minimums[1]]:
            ret[i] = 1 + np.arcsin(np.sqrt(ii0[i]))/np.pi
        elif vrms[i] <= vrms[maximums[2]]:
            ret[i] = 1 - np.arcsin(np.sqrt(ii0[i]))/np.pi
        else:
            ret[i] = np.arcsin(np.sqrt(ii0[i]))/np.pi
    return vrms, ret
