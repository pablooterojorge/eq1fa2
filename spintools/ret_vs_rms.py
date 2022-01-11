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
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=False, figsize=(12,5)) 
    ax1.set_xlabel('$\Delta V_{rms}$ [V]')
    ax1.set_ylabel('d $ \Delta n $ / $\lambda $')
    ax1.plot(vrms, ret, '*k')
    ax2.set_xlabel('d $ \Delta n $ / $\lambda $')
    ax2.set_ylabel(r'I/$ I_0 $')
    ax2.plot(ret, ii0, 'ok')
    return vrms, ret
