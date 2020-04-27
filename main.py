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
                 kde=True,
                 hist=False,
                 fit=scipy.stats.norm,
                 label=str(name),
                 ax=axis,
                 color=color)
    pass

def hist_plot_laplace(data, name, axis,color):
    sns.distplot(data,
                 kde=True,
                 hist=False,
                 fit=scipy.stats.laplace,
                 label=str(name),
                 ax=axis,
                 color=color)
    pass

def hist_plot_uniform(data, name, axis,color):
    sns.distplot(data,
                 kde=True,
                 hist=False,
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
f, axs = plt.subplots(nrows=len(dfs),ncols=1,figsize=(8,(len(dfs)+1)*8/3))

i=0
for ax in axs.reshape(-1):
    hist_plot(dfs[i].PctChg,names[i],ax,colors[i])
    ax.legend(loc="upper left")
    mean = dfs[i].PctChg.mean()
    std = dfs[i].PctChg.std()
    max = dfs[i].PctChg.max()
    min = dfs[i].PctChg.min()
    ax.text(0.99,
            0.95,
            r'$\mu$ = '+str(mean)[0:6]+
            '%\n$\sigma$ = '+str(std)[0:4]+'%'+
            '\nmax = '+str(max)[0:6]+'%'+
            '\nmin = '+str(min)[0:6]+'%',
            ha='right',
            va='top',
            fontsize=9,
            transform=ax.transAxes,
            color=colors[i])
    ax.set_xlabel('')
    ax.set_ylabel('Frequency',fontsize=8)
    #ax.set(yscale='log')
    i+=1

plt.tight_layout()
#plt.suptitle("% Daily Change Distribution")
#fig('Daily_Dist.png',dpi=300)

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
            hist_plot(dfs[row].PctChg, names[row], axs[row,col], colors[row])
            hist_plot(norm_val[row],names[row], axs[row,col],'r')
            if row == 0:
                axs[row,col].set_title('Normal Distribution')
        if col == 1:
            hist_plot(dfs[row].PctChg, names[row], axs[row,col], colors[row])
            hist_plot(laplace_val[row],names[row], axs[row,col],'r')
            if row == 0:
                axs[row,col].set_title('Laplace Distribution')
        if col == 2:
            hist_plot(dfs[row].PctChg, names[row], axs[row,col], colors[row])
            hist_plot(uniform_val[row],names[row], axs[row,col],'r')
            if row == 0:
                axs[row,col].set_title('Uniform Distribution')

        axs[row,col].set_ylabel('Frequency', fontsize=8)
        axs[row,col].set_xlabel('')
        axs[row,col].get_legend().remove()
        #axs[row,col].set(yscale='log')

plt.tight_layout()
plt.savefig('Distribution_Plots.png',dpi=600)

# plot Normal and Laplace distribution plots in log space
sns.set(color_codes=True)
f, axs = plt.subplots(nrows=len(dfs),ncols=2,figsize=(12,(len(dfs)+1)*8/3))
for row in range(len(dfs)):
    for col in range(2):
        if col == 0:
            hist_plot(dfs[row].PctChg, names[row], axs[row,col], colors[row])
            hist_plot(norm_val[row],names[row], axs[row,col],'r')
            if row == 0:
                axs[row,col].set_title('Normal Distribution')
        if col == 1:
            hist_plot(dfs[row].PctChg, names[row], axs[row,col], colors[row])
            hist_plot(laplace_val[row],names[row], axs[row,col],'r')
            if row == 0:
                axs[row,col].set_title('Laplace Distribution')

        axs[row,col].set_ylabel('Frequency', fontsize=8)
        axs[row,col].set_xlabel('')
        axs[row,col].set(yscale='log')
        axs[row,col].get_legend().remove()

plt.tight_layout()
plt.savefig('Log_Distribution_Plots.png',dpi=600)

# --------------------------------------------------------------------------------------------


# ---------- MONTE CARLO SIMULATION --------------------------------
runs=1000
days=90

f, axs = plt.subplots(nrows=len(dfs),ncols=1,figsize=((len(dfs)+1)*8/3,8))

for i in range(len(dfs)):

    # --- generate random number list based on normal stock distribution
    norm_val = np.random.laplace(loc=dfs[i].PctChg.mean(), scale=dfs[i].PctChg.std(), size=(runs, days))

    df1 = dfs[i][dfs[i].Date >= pd.to_datetime('4/1/2019')]
    sns.lineplot(df1.Date, df1.Open, ax=axs[i],color=colors[i],label=names[i])

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

    for line in axs[i].lines[1:]:
        line.set_linestyle("--")
        line.set_linewidth(0.5)

plt.tight_layout()
plt.legend()
plt.savefig('future_'+str(runs)+'.png',dpi=600)

finish = datetime.datetime.now()
print(finish-start)
print(f'Completed in: {finish-start} sec.')

#plt.show()
