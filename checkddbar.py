import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt

from ROOT import TH1F, TF1, TCanvas
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count

import lz4.frame
import time

debug = True
#debug = False

start= time.time()
dfreco1 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco2_4_0.75.pkl.lz4", "rb"))
dfreco2 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))
dfreco3 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco6_8_0.65.pkl.lz4", "rb"))
#dfreco4 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco8_24_0.65.pkl.lz4", "rb"))
frames = [dfreco1, dfreco2, dfreco3]
dfreco = pd.concat(frames)
end = time.time()
print("Data loaded in", end - start, "sec")

dfreco = dfreco.query("y_test_probxgboost>0.8")

if(debug):
    print("Debug mode: reduced events")
    dfreco = dfreco[:20000]
print("Size of data", dfreco.shape)

binning = 200
fit_fun1 = TF1("fit_fun1", "expo" ,1.64, 1.82)
fit_fun2 = TF1("fit_fun2", "gaus", 1.82, 1.92)
fit_total = TF1("fit_total", "expo(0) + gaus(2) + expo(5)", 1.64, 2.1)
cYields = TCanvas('cYields', 'The Fit Canvas')
h_invmass = TH1F("invariant mass" , "", binning, dfreco.inv_mass.min(),
        dfreco.inv_mass.max())
fill_hist(h_invmass, dfreco.inv_mass)
h_invmass.Fit(fit_fun1, "R")
par1 = fit_fun1.GetParameters()
print("parameters for expo", par1[0], par1[1])
h_invmass.Fit(fit_fun2, "R+")
par2 = fit_fun2.GetParameters()
print("parameters for gaus", par2[0], par2[1], par2[2])
fit_total.SetParameters(par1[0], par1[1], par2[0], par2[1], par2[2], par1[0],par1[1])
h_invmass.Fit(fit_total,"R+")
par = fit_total.GetParameters()
h_invmass.Draw()
cYields.SaveAs("h_invmass.pdf")

h_pt_prong0 = TH1F("pt prong_0" , "", 200, 0.8, 8.)
fill_hist(h_pt_prong0, dfreco.pt_prong0)
h_pt_prong0.Draw()
cYields.SaveAs("h_pt_prong0.pdf")

h_pt_prong1 = TH1F("pt prong_1" , "", 200, 0.8, 8.)
fill_hist(h_pt_prong1, dfreco.pt_prong1)
h_pt_prong1.Draw()
cYields.SaveAs("h_pt_prong1.pdf")

h_eta_prong0 = TH1F("eta prong_0" , "", 200, 0., 0.9)
fill_hist(h_eta_prong0, dfreco.eta_prong0)
h_eta_prong0.Draw()
cYields.SaveAs("h_eta_prong0.pdf")

h_eta_prong1 = TH1F("eta prong_1" , "", 200, 0., 0.9)
fill_hist(h_eta_prong1, dfreco.eta_prong1)
h_eta_prong1.Draw()
cYields.SaveAs("h_eta_prong1.pdf")

h_eta_cand = TH1F("eta cand" , "", 200, 0., 0.9)
fill_hist(h_eta_cand, dfreco.eta_cand)
h_eta_cand.Draw()
cYields.SaveAs("h_eta_cand.pdf")

h_phi_cand = TH1F("phi cand" , "", 200, 0., 8.)
fill_hist(h_phi_cand, dfreco.phi_cand)
h_phi_cand.Draw()
cYields.SaveAs("h_phi_cand.pdf")

h_pt_cand = TH1F("pt cand" , "", 200, 2., 8.)
fill_hist(h_pt_cand, dfreco.pt_cand)
h_pt_cand.Draw()
cYields.SaveAs("h_pt_cand.pdf")

#lets try to do groupby as parallelized function over the dataframe
start = time.time()
grouped = dfreco.groupby(["run_number","ev_id"])
end = time.time()
print("groupby done in", end - start, "sec")

num_cores = int(cpu_count()/2)
num_part  = num_cores*2

print("start parallelizing with", num_cores, "cores")
def parallelize_df(df, func):
    df_split = np.array_split(df, num_part)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

start = time.time()
inv_range = dfreco.inv_mass.max() - dfreco.inv_mass.min()
end = time.time()
print("inv_range", inv_range, "done in", start-end, "sec")
bin_size = inv_range/binning
end2 = time.time()
bins_arr = [bins for bins in np.arange (dfreco.inv_mass.min(),
        dfreco.inv_mass.max(), bin_size)]
print("bins array created in", end2-end, "sec")

#idx = pd.cut(dfreco.inv_mass, bins=np.arange(dfreco.inv_mass.min(), dfreco.inv_mass.max(), bin_size),
#                      include_lowest=True, right=False)
#groups = dfreco.groupby(idx, as_index = False)
#for name, group in groups:
#    result = np.exp(par[0] + par[1]*group.inv_mass.mean())
#    print("group_len", len(group), "inv_mass mean", group.inv_mass.mean(), "result of grop thing", result)

#grouped_inv = dfreco.groupby(idx, as_index = False)
#print("invmass grouped")
#for name, group in grouped_inv:
#    print(len(group))

def filter_invmass(df):
  idx = pd.cut(df.inv_mass, bins=np.arange(dfreco.inv_mass.min(), dfreco.inv_mass.max(), bin_size),
                      include_lowest=True, right=False)
  groups = df.groupby(idx, as_index = False)
  for name, group in groups:
      print(len(group))
  df = df.groupby(idx, as_index = False).filter(lambda x: len(x) - np.exp(par[0] +
        par[1] * x.inv_mass.mean()) > 0)
  print(df.describe())
  return df

def filter_eta(df):
    df = df.groupby(["run_number", "ev_id"], sort = False).filter(lambda x:
            x.eta_cand.max() - x.eta_cand.min() > 0)
    return df

def filter_phi(df):
    df = df.groupby(["run_number", "ev_id"], sort = False).filter(lambda x:
            x.phi_cand.max() - x.phi_cand.min() > 0)
    return df

start = time.time()
filtrated_invmass = parallelize_df(dfreco, filter_invmass)
end = time.time()
print("invmass filter", end - start, "sec")
inv_vec = filtrated_invmass["inv_mass"]
print("inv_vec", inv_vec)
#filtrated_invmass = filtrated_invmass.groupby(["inv_mass"])
cYields = TCanvas('cYields', 'The Fit Canvas')
h_invmass_filtr = TH1F("invariant mass filtered" , "", binning,
        dfreco.inv_mass.min(), dfreco.inv_mass.max())
print("vector of invarian masses", inv_vec)
fill_hist(h_invmass_filtr, inv_vec)
h_invmass_filtr.Draw()
cYields.SaveAs("h_invmass_filtr.pdf")

input()
start = time.time()
filtrated_eta = parallelize_df(dfreco, filter_eta)
end = time.time()
print("eta filter", end - start, "sec")
filtrated_phi = parallelize_df(filtrated_invmass, filter_phi)
end2 = time.time()
print("phi filter", end2 - end, "sec")
print("paralellizing is done in", end2 - start, "sec")

start = time.time()
grouplen = pd.array(grouped.size())
end = time.time()
print("creating grouplen array", end - start, "sec")
h_grouplen = TH1F("group_length" , "", 5, 1., 6.)
fill_hist(h_grouplen, grouplen)
h_grouplen.Draw()
cYields.SaveAs("h_grouplen.pdf")

#filtrated_phi = grouped.filter(lambda x: x.phi_cand.max() - x.phi_cand.min() >
#        0).groupby(["run_number", "ev_id"])
#filtrated_eta = grouped.filter(lambda x: x.eta_cand.max() - x.eta_cand.min() >
#        0).groupby(["run_number", "ev_id"])

#print("Grouped and filtered")

start = time.time()
filtrated_phi = filtrated_phi.groupby(["run_number", "ev_id"])
end1 = time.time()
phi_vec     = filtrated_phi["phi_cand"]
d_phi_dist = np.abs(phi_vec.max() - phi_vec.min())
end2 = time.time()
print("grouping phi", end1 - start, "sec")
print("calc dist", end2 - end1, "sec")
h_d_phi_cand = TH1F("delta phi cand" , "", 200, 0., 5.)
fill_hist(h_d_phi_cand, d_phi_dist)
cYields.SetLogy(True)
h_d_phi_cand.Draw()
cYields.SaveAs("h_d_phi_cand.pdf")

start = time.time()
filtrated_eta = filtrated_eta.groupby(["run_number", "ev_id"])
end1 = time.time()
eta_vec     = filtrated_eta["eta_cand"]
d_eta_dist = np.abs(eta_vec.max() - eta_vec.min())
end2 = time.time()
print("grouping eta", end1 - start, "sec")
print("calc dist", end2 - end1, "sec")
h_d_eta_cand = TH1F("delta eta cand" , "", 200, 0., 3.)
fill_hist(h_d_eta_cand, d_eta_dist)
h_d_eta_cand.Draw()
cYields.SaveAs("h_d_eta_cand.pdf")
