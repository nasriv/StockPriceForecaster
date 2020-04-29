import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np
import datetime

'''
Use probabilistic distributions to determine future prices based on past volatility 
'''

start = datetime.datetime.now()
# ---------------------------------------------------
def import_df(file):
    df = pd.read_csv(file)
    df.Date = pd.to_datetime(df['Date'])
    df['PctChg'] = (df.Open - df.Close).div(df.Open)*100
    return df

def hist_plot(data, name, axis,color):
    sns.distplot(data,
                 kde=True,
                 hist=False,
                 label=str(name),
                 ax=axis,
                 color=color)
    pass

def hist_plot_norm(data, name, axis,color):
    sns.distplot(data,
                 kde=False,
                 hist=True,
                 fit=scipy.stats.norm,
                 label=str(name),
                 ax=axis,
                 color=color)
    pass

def hist_plot_laplace(data, name, axis,color):
    sns.distplot(data,
                 kde=False,
                 hist=True,
                 fit=scipy.stats.laplace,
                 label=str(name),
                 ax=axis,
                 color=color)
    pass

def hist_plot_uniform(data, name, axis,color):
    sns.distplot(data,
                 kde=False,
                 hist=True,
                 fit=scipy.stats.uniform,
                 label=str(name),
                 ax=axis,
                 color=color)
    pass

def line_plot(data,name,axis,color):
    sns.lineplot(data.Date,
                 data.Open,
                 label=str(name),
                 ax=axis,
                 color=color)
    pass
# ---------------------------------------------------

# Create dataframes ---------------------
SPY = import_df('SPY.csv')
AMD = import_df('AMD.csv')

# Define dataframe parameter lists for looping ---------------------

names = ['SPDR S&P500 ETF',
         'Advanced Micro Devices']
dfs = [SPY, AMD]
colors = ['b','green','r','orange']

# -------- HISTOGRAM PERCENT DISTRIBUTION PLOT-------------
sns.set(color_codes=True)

f, axs = plt.subplots(nrows=1,ncols=1,figsize=(8,(len(dfs)+1)*8/3))

i=0
for i in range(len(dfs)):
    sns.distplot(dfs[i].PctChg,kde=True,label=names[i],ax=axs,color=colors[i])
    axs.legend(loc="upper left")
    mean = dfs[i].PctChg.mean()
    std = dfs[i].PctChg.std()
    max = dfs[i].PctChg.max()
    min = dfs[i].PctChg.min()
    if i == 0:
        xloc = 0.99
        yloc = 0.95
        halign='right'
    else:
        xloc = 0.99
        yloc = 0.75
        halign='right'
    axs.text(xloc,
            yloc,
            r'$\mu$ = '+str(mean)[0:6]+
            '%\n$\sigma$ = '+str(std)[0:4]+'%'+
            '\nmax = '+str(max)[0:6]+'%'+
            '\nmin = '+str(min)[0:6]+'%',
            ha=halign,
            va='top',
            fontsize=9,
            transform=axs.transAxes,
            color=colors[i])
    axs.set_xlabel('% Change')
    axs.set_ylabel('Probability Density')
    #ax.set(yscale='log')
    i+=1

plt.tight_layout()
plt.title('Daily % Return Probability Distribution of AMD and SPY')
plt.savefig('Daily%Dist.png', dpi=600)
#----- Random distributions -----------------------------

norm_val=[None]*(len(dfs)+1) # initialize list to number of stocks
laplace_val=[None]*(len(dfs)+1) # initialize list to number of stocks
uniform_val=[None]*(len(dfs)+1) # initialize list to number of stocks


for i in range(len(dfs)):
    (mu_N, std_N) = scipy.stats.norm.fit(dfs[i].PctChg)
    (mu_L, std_L) = scipy.stats.laplace.fit(dfs[i].PctChg)

    norm_val[i]=np.random.normal(loc=mu_N,
                              scale=std_N,
                              size=(len(dfs[i]),1))
    laplace_val[i]=np.random.laplace(loc=mu_L,
                              scale=std_L,
                              size=(len(dfs[i]),1))
    uniform_val[i]=np.random.uniform(low=dfs[i].PctChg.min(),
                              high=dfs[i].PctChg.max(),
                              size=(len(dfs[i]),1))

# plot all three distribution plots in linear space
sns.set(color_codes=True)
f, axs = plt.subplots(nrows=len(dfs),ncols=3,figsize=(12,(len(dfs)+1)*8/3))
for row in range(len(dfs)):
    for col in range(3):
        if col == 0:
            res = scipy.stats.probplot(dfs[row].PctChg,dist='norm',fit=True,plot=axs[row][col])
            (slope, intercept, r) = res[1]
            axs[row, col].text(0.85, 0.15,
                               'Good Fit',
                               transform=axs[row, col].transAxes,
                               size=10,
                               ha='right',
                               bbox=dict(fc='white', alpha=0.6))
            if row == 0:
                axs[row,col].set_title('Normal Distribution')
                axs[row, col].get_lines()[0].set_markerfacecolor('b')
            else:
                axs[row,col].set_title('')
                axs[row, col].get_lines()[0].set_markerfacecolor('green')
        if col == 1:
            res = scipy.stats.probplot(dfs[row].PctChg,dist='laplace',fit=True,plot=axs[row][col])
            (slope, intercept, r) = res[1]
            axs[row, col].text(0.85, 0.15,
                               'Best Fit',
                               transform=axs[row, col].transAxes,
                               size=10,
                               ha='right',
                               bbox=dict(fc='white', alpha=0.6))
            if row == 0:
                axs[row,col].set_title('Laplace Distribution')
            else:
                axs[row,col].set_title('')
        if col == 2:
            res = scipy.stats.probplot(dfs[row].PctChg,dist='uniform',fit=True,plot=axs[row][col])
            (slope, intercept, r) = res[1]
            axs[row, col].text(0.85, 0.15,
                               'Poor Fit',
                               transform=axs[row, col].transAxes,
                               size=10,
                               ha='right',
                               bbox=dict(fc='white', alpha=0.6))
            if row == 0:
                axs[row,col].set_title('Uniform Distribution')
            else:
                axs[row,col].set_title('')

        axs[row,col].set_xlabel('')
        axs[row,col].set_ylabel('')
        axs[row,col].get_lines()[0].set_markersize(1.0)
        axs[row,col].text(0.05, 0.85,
                          str(names[row])+'\n'+r'$r^{2} =$' + str(r ** 2)[0:5],
                          transform=axs[row,col].transAxes,
                          size=10,
                          bbox=dict(fc='white',alpha=0.6))


plt.tight_layout()
plt.savefig('QQ_Plots.png',dpi=600)

# --------------------------------------------------------------------------------------------


# ---------- MONTE CARLO SIMULATION --------------------------------
runs=1000
days=90

# ---- track final price for each stock ---
results = []
f, axs = plt.subplots(nrows=len(dfs),ncols=1,figsize=((len(dfs)+1)*8/3,8))

for i in range(len(dfs)):

    # --- generate random number list based on normal stock distribution
    norm_val = np.random.laplace(loc=dfs[i].PctChg.mean(), scale=dfs[i].PctChg.std(), size=(runs, days))

    df1 = dfs[i][dfs[i].Date >= pd.to_datetime('4/1/2019')]
    sns.lineplot(df1.Date, df1.Open, ax=axs[i],color=colors[i],label=names[i])


    result_temp = []

    for run in range(runs):
        temp = dfs[i]
        for day in range(days):
            df2 = pd.DataFrame(columns=temp.columns, index=[0])
            df2.iloc[0]['Date'] = temp.loc[temp.index[-1]]['Date'] + datetime.timedelta(days=1)
            df2.iloc[0]['PctChg'] = float(norm_val[run][day])
            df2.iloc[0]['Open'] = (temp.loc[temp.index[-1]]['Open'] * float(1 + norm_val[run][day] / 100))
            df2.Date = pd.to_datetime(df2.Date)
            temp = pd.concat([temp, df2], ignore_index=True)
            temp.Open = temp.Open.astype(float)

        df2 = temp[temp.Date >= pd.to_datetime('4/14/2020')]

        sns.lineplot(df2.Date,df2.Open,ax=axs[i],alpha=0.5,color=colors[i])
        axs[i].set_xlabel("")

        result_temp.append(float(df2.tail(1).Open.values))

    results.append(result_temp)

    for line in axs[i].lines[1:]:
        line.set_linestyle("--")
        line.set_linewidth(0.5)

    axs[i].set_ylabel('Share Price ($)')

plt.tight_layout()
plt.legend()
plt.savefig('future_'+str(runs)+'.png',dpi=600)


# ---- plot end price results ----
f, axs = plt.subplots(2, 1, figsize=(12,8))

axs[0].axvline(287,0,1,color='r',linestyle='--',label='Current Share Price')
#axs[0].axvline(sum(results[0])/len(results[0]),0,1,color='b',linestyle='--',label='Result Average')
axs[1].axvline(56.63,0,1,color='r',linestyle='--',label='Current Share Price')
#axs[1].axvline(sum(results[1])/len(results[1]),0,1,color='green',linestyle='--',label='Result Average')

i=0
for ax in axs.reshape(-1):
    sns.distplot(results[i],hist=True,kde=True,ax=ax,label=names[i],color=colors[i])
    ax.legend(loc="best")
    ax.set_xlabel('Share Price ($)')
    ax.set_ylabel('Probability Density', fontsize=10)
    i+=1

plt.tight_layout()
plt.savefig('EndPriceData_'+str(runs)+'.png',dpi=600)

# ------------------------
finish = datetime.datetime.now()
print(finish-start)
print(f'Completed in: {finish-start} sec.')

#plt.show()
