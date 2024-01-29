def init():
    init_dict = {
        'xlabel': r'wavenumbers $cm^{-1}$',
        'ylabel': r'$\Delta$ abs. [mOD]',
        'xhigh': 4000,
        'xlow' : 1000,
        'yhigh': 1500 ,
        'ylow' : -500,
        'ymulti':1000
        }

    return init_dict

def initialprojectdict():
    projectdict = {'no files':0}
    return projectdict

def j_son_spectrum(x,y,bg,bgkey,bgscale,show,col,lwidth,lstyle,lab,subpl):
    dataset = {'xdata': x,
                    'ydata': y,
                    'bg': False,
                    'bgkey':bgkey,
                    'bgscale': bgscale,
                    'show': show,       #defaults
                    'color': col,    #...
                    'linewidth': lwidth,
                    'linestyle':lstyle,
                    'label':lab ,
                    'subplot': subpl         
                            }
    return dataset