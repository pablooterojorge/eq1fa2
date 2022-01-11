def fitret (params,x, y):
    trans = params[‘trans’]
    wl = params[‘wl’]
    d = params[‘d’]
    deltan = params[‘deltan’]
    fitii0 = np.sin(np.pi* deltan*d / wl)
    return (fitii0 - y)
