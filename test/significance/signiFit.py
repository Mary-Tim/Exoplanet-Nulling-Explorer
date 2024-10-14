import h5py
import numpy as np
import matplotlib.pyplot as plt
import tabulate as tab
from scipy import stats
from iminuit import Minuit
import mplhep
plt.style.use(mplhep.style.LHCb2)

def fitChi2Hist(data, pdf, **kwargs): 
    data = np.array(data)
    def nll(df):
        return -np.sum(np.log(pdf(data, df)))
    nll.errordef = Minuit.LIKELIHOOD
    nll.error_df=0.1
    m = Minuit(nll, df=2)
    m.migrad()
    m.hesse()
    return m

def fitChi2scHist(data, pdf, **kwargs): 
    data = np.array(data)
    def nll(df, scale):
        return -np.sum(np.log(pdf(data, df, 0, scale)))
    nll.errordef = Minuit.LIKELIHOOD
    nll.error_df=0.1
    nll.error_scale=0.1
    m = Minuit(nll, df=2, scale=1)
    m.migrad()
    m.hesse()
    return m

def Pull():
    import os
    import uproot
    import math

    #path = "results/Signi_20240726_111904"
    path = "results/Signi_20240916_231055"
    file= h5py.File(f"{path}/toy_nll_distribution.hdf5")
    bkg_nll = file['bkg_nll_array'][:]
    sig_nll = file['sig_nll_array'][:]

    d2NLLlist = 2*(bkg_nll-sig_nll)

    Ndata = len(d2NLLlist)
    print("Sample amount:", Ndata)
    df, loc, scale = stats.chi2.fit(d2NLLlist,floc=0)#,fscale=1)
    print("##### Fitting with scipy.stats.chi2.fit")
    print(f"df={df}\tloc={loc}\tscale={scale}")
    print("##### Fitting with iminuit")
    msc = fitChi2scHist(d2NLLlist, stats.chi2.pdf)
    print(tab.tabulate(*msc.params.to_table()))
    #df = msc.values["df"]; loc=0; scale=msc.values["scale"]
    print("##### When fixing scale (which is used in the plot)")
    m = fitChi2Hist(d2NLLlist, stats.chi2.pdf)
    print(tab.tabulate(*m.params.to_table()))
    df = m.values["df"]; loc=0; scale=1
    Nbins = 10 # number of bins (edit here)
    plt.figure(figsize=[12,8])
    _, bins, _ = plt.hist(d2NLLlist, bins=Nbins, alpha=0.6, label="Toy 2ΔLL distribution")
    binwidth = np.mean(bins[1:]-bins[:-1])
    xx = np.linspace(0,80,500) # x range of overlaid chi2 curve (edit here)
    plt.plot(xx, Ndata*binwidth*stats.chi2.pdf(xx,df=df,loc=loc,scale=scale),'r', label="Toy 2ΔLL fitted")

    #d2NLL = 2*abs(14870.324-15013.886) # Δ2NLL from real data cfit (edit here)
    d2NLL = 2*abs(10774.024-10946.437603954346) # Δ2NLL from real data cfit (edit here)

    stick_height = 20 # the height of vertical bars in the plot (edit here)
    plt.plot(np.ones(2)*stats.chi2.ppf(0.997,df=df,loc=loc,scale=scale),np.linspace(0,stick_height,2),'r--', label="99.7% limit (3σ)")
    plt.plot(np.ones(2)*stats.chi2.ppf(0.9999994,df=df,loc=loc,scale=scale),np.linspace(0,stick_height,2),'r:', label="99.99994% limit (5σ)")
    plt.plot(np.ones(2)*d2NLL,np.linspace(0,stick_height,2),'purple', label="Data 2ΔLL")
    plt.legend(fontsize='xx-large', loc='upper center')
    plt.xlabel("2ΔLL", fontsize='xx-large', position=[0.95,0])
    plt.ylabel("Entries / ({:.2})".format(binwidth), fontsize='xx-large', position=[0,0.85])
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    #plt.title(f"Toy 2ΔLL distribution", fontsize='xx-large')
    plt.savefig(f"{path}/sigtestZ0_.pdf") # save path (edit here)
    print("Significance: ", -stats.norm.ppf(stats.chi2.sf(d2NLL,df=df,loc=loc,scale=scale)/2))
    print(f"ndf: {df} +/- {m.errors['df']}")
    plt.show()

def main():
    Pull()

if __name__ == "__main__":
    main()
