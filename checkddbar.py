import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt

from ROOT import TH1F, TH2F, TF1, TCanvas
from ROOT import kBlack, kBlue, kRed, kGreen, kMagenta, TLegend
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count

import lz4.frame
import time

#debug = True
debug = False

plots = True
#plots = False

make_phi_compare = True
#make_phi_compare = False

d_phi_cut = 0.

start= time.time()

if (debug):
    dframe = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))

else:
    dfreco0 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco1_2_0.75.pkl.lz4", "rb"))
    dfreco1 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco2_4_0.75.pkl.lz4", "rb"))
    dfreco2 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))
    dfreco3 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco6_8_0.65.pkl.lz4", "rb"))
    dfreco4 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco8_24_0.45.pkl.lz4", "rb"))
    frames = [dfreco0, dfreco1, dfreco2, dfreco3, dfreco4]
    dframe = pd.concat(frames)

dframe = dframe.query("y_test_probxgboost>0.5")
#dframe = dframe.query("pt_cand > 4")
#dframe = dframe.query("pt_cand < 10")
dfreco = dframe.reset_index(drop = True)

end = time.time()
print("Data loaded in", end - start, "sec")

if(debug):
    print("Debug mode: reduced data")
    dfreco = dfreco[:5000]
print("Size of data", dfreco.shape)

print(dfreco.columns)

binning = 200

cYields = TCanvas('cYields', 'The Fit Canvas')
fit_fun1 = TF1("fit_fun1", "expo" ,1.64, 1.82)
fit_fun2 = TF1("fit_fun2", "gaus", 1.82, 1.92)
fit_total = TF1("fit_total", "expo(0) + gaus(2) + expo(5)", 1.64, 2.1)
h_invmass = TH1F("invariant mass" , "", binning, dfreco.inv_mass.min(),
        dfreco.inv_mass.max())
fill_hist(h_invmass, dfreco.inv_mass)
h_invmass.Fit(fit_fun1, "R")
par1 = fit_fun1.GetParameters()
h_invmass.Fit(fit_fun2, "R+")
par2 = fit_fun2.GetParameters()
fit_total.SetParameters(par1[0], par1[1], par2[0], par2[1], par2[2],
        par1[0],par1[1])
h_invmass.Fit(fit_total,"R+")
par = fit_total.GetParameters()
h_invmass.Draw()
cYields.SaveAs("h_invmass.png")

if (plots):
    h_d_len = TH1F("d_len" , "", 200, dfreco.d_len.min(), dfreco.d_len.max())
    fill_hist(h_d_len, dfreco.d_len)
    h_d_len.Draw()
    cYields.SaveAs("h_d_len.png")

    h_norm_dl = TH1F("norm dl" , "", 200, dfreco.norm_dl.min(), dfreco.norm_dl.max())
    fill_hist(h_norm_dl, dfreco.norm_dl)
    h_norm_dl.Draw()
    cYields.SaveAs("h_norm_dl.png")

    h_cos_p = TH1F("cos_p" , "", 200, dfreco.cos_p.min(), dfreco.cos_p.max())
    fill_hist(h_cos_p, dfreco.cos_p)
    h_cos_p.Draw()
    cYields.SaveAs("h_cos_p.png")

    h_nsigTPC_K_0 = TH1F("nsigma TPC K_0" , "", 200, dfreco.nsigTPC_K_0.min(),
            dfreco.nsigTPC_K_0.max())
    print(dfreco.nsigTPC_K_0.min(), dfreco.nsigTPC_K_0.max())
    fill_hist(h_nsigTPC_K_0, dfreco.nsigTPC_K_0)
    h_nsigTPC_K_0.Draw()
    cYields.SaveAs("nsigTPC_K_0.png")

    h_nsigTPC_K_1 = TH1F("nsigTPC_K_1 " , "", 200, dfreco.nsigTPC_K_1.min(),
            dfreco.nsigTPC_K_1.max())
    fill_hist(h_nsigTPC_K_1, dfreco.nsigTPC_K_1)
    h_nsigTPC_K_1.Draw()
    cYields.SaveAs("h_nsigTPC_K_1.png")

    h_nsigTOF_K_0 = TH1F("nsigma TOF K_0" , "", 200, dfreco.nsigTOF_K_0.min() ,
            dfreco.nsigTOF_K_0.max() )
    fill_hist(h_nsigTOF_K_0 , dfreco.nsigTOF_K_0 )
    h_nsigTOF_K_0.Draw()
    cYields.SaveAs("nsigTOF_K_0.png")

    h_nsigTOF_K_1 = TH1F("nsigTOF_K_1 " , "", 200, dfreco.nsigTOF_K_1.min(),
            dfreco.nsigTOF_K_1.max())
    fill_hist(h_nsigTOF_K_1, dfreco.nsigTOF_K_1)
    h_nsigTOF_K_1.Draw()
    cYields.SaveAs("h_nsigTOF_K_1.png")

    h_pt_prong0 = TH1F("pt prong_0" , "", 200,  dfreco.pt_prong0.min(),
            dfreco.pt_prong0.max())
    fill_hist(h_pt_prong0, dfreco.pt_prong0)
    h_pt_prong0.Draw()
    cYields.SaveAs("h_pt_prong0.png")

    h_pt_prong1 = TH1F("pt prong_1" , "", 200,  dfreco.pt_prong1.min(),
            dfreco.pt_prong1.max())
    fill_hist(h_pt_prong1, dfreco.pt_prong1)
    h_pt_prong1.Draw()
    cYields.SaveAs("h_pt_prong1.png")

    h_eta_prong0 = TH1F("eta prong_0" , "", 200, dfreco.eta_prong0.min(),
            dfreco.eta_prong0.max())
    fill_hist(h_eta_prong0, dfreco.eta_prong0)
    h_eta_prong0.Draw()
    cYields.SaveAs("h_eta_prong0.png")

    h_eta_prong1 = TH1F("eta prong_1" , "", 200, dfreco.eta_prong1.max(),
            dfreco.eta_prong1.max())
    fill_hist(h_eta_prong1, dfreco.eta_prong1)
    h_eta_prong1.Draw()
    cYields.SaveAs("h_eta_prong1.png")

    h_eta_cand = TH1F("eta cand" , "", 200, dfreco.eta_cand.min(),
            dfreco.eta_cand.max())
    fill_hist(h_eta_cand, dfreco.eta_cand)
    h_eta_cand.Draw()
    cYields.SaveAs("h_eta_cand.png")

    h_phi_cand = TH1F("phi cand" , "", 200, dfreco.eta_cand.min(),
            dfreco.eta_cand.max())
    fill_hist(h_phi_cand, dfreco.phi_cand)
    h_phi_cand.Draw()
    cYields.SaveAs("h_phi_cand.png")

    h_pt_cand = TH1F("pt cand" , "", 200, dfreco.pt_cand.min(),
            dfreco.pt_cand.max())
    fill_hist(h_pt_cand, dfreco.pt_cand)
    h_pt_cand.Draw()
    cYields.SaveAs("h_pt_cand.png")

start = time.time()
grouped = dfreco.groupby(["run_number","ev_id"])
end = time.time()
print("groupby done in", end - start, "sec")
start = time.time()
grouplen = pd.array(grouped.size())
end = time.time()
gmin = grouplen.min()
gmax = grouplen.max()
g_bins = gmax - gmin
print("creating grouplen array", end - start, "sec")
h_grouplen = TH1F("group_length" , "", int(g_bins), gmin, gmax)
fill_hist(h_grouplen, grouplen)
cYields.SetLogy(True)
h_grouplen.Draw()
cYields.SaveAs("h_grouplen.png")

#start = time.time()
#inv_range = dfreco.inv_mass.max() - dfreco.inv_mass.min()
#end = time.time()
#print("inv_range", inv_range, "done in", start-end, "sec")
#bin_size = inv_range/binning
#end2 = time.time()
#bins_arr = [bins for bins in np.arange (dfreco.inv_mass.min(),
#        dfreco.inv_mass.max(), bin_size)]
#print("bins array created in", end2-end, "sec")
#
#idx = pd.cut(dfreco.inv_mass, bins=np.arange(dfreco.inv_mass.min(), dfreco.inv_mass.max(), bin_size),
#                      include_lowest=True, right=False)


#parallelized functions over the dataframe

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

def filter_eta(df):
    df = df.groupby(["run_number", "ev_id"], sort = False).filter(lambda x:
            x.eta_cand.max() - x.eta_cand.min() > 0)
    return df

def filter_phi(df):
    df = df.groupby(["run_number", "ev_id"], sort = False).filter(lambda x:
            x.phi_cand.max() - x.phi_cand.min() > d_phi_cut)
    return df

start = time.time()
#filtrated_eta = parallelize_df(dfreco, filter_eta)
#
#end = time.time()
#print("eta filter", end - start, "sec")

filtrated_phi = parallelize_df(dfreco, filter_phi)

end2 = time.time()
#print("phi filter", end2 - end "sec")

print("paralellizing is done in", end2 - start, "sec")

#filtrated_phi = grouped.filter(lambda x: x.phi_cand.max() - x.phi_cand.min() >
#        0).groupby(["run_number", "ev_id"])
#filtrated_eta = grouped.filter(lambda x: x.eta_cand.max() - x.eta_cand.min() >
#        0).groupby(["run_number", "ev_id"])

#print("Grouped and filtered")



grouped_phi = filtrated_phi.groupby(["run_number", "ev_id"])
phi_vec     = grouped_phi["phi_cand"]

d_phi_dist = np.abs(phi_vec.max() - phi_vec.min()) #delta phi destribution as difference between max and min phi in the group

# find positions of max and min elements in group
min_idx_vec = phi_vec.idxmin()
max_idx_vec = phi_vec.idxmax()

new_df_min = filtrated_phi.loc[min_idx_vec,]
print("compare length of min elements amount and length of new df",
        len(min_idx_vec), len(new_df_min))

new_df_max = filtrated_phi.loc[max_idx_vec,]
print("compare length of min elements amount and length of new df",
        len(max_idx_vec), len(new_df_max))

eta_max_vec = new_df_max["eta_cand"]
eta_min_vec = new_df_min["eta_cand"]

inv_mass_max_vec = new_df_max["inv_mass"]
inv_mass_min_vec = new_df_min["inv_mass"]

pt_max_vec = new_df_max["pt_cand"]
pt_min_vec = new_df_min["pt_cand"]

#h_d_phi_cand = TH1F("delta phi cand" , "", 200, d_phi_dist.min(),
#        d_phi_dist.max())
#fill_hist(h_d_phi_cand, d_phi_dist)
#cYields.SetLogy(True)
#h_d_phi_cand.Draw()
#cYields.SaveAs("h_d_phi_cand.png")

if (make_phi_compare):
    cYields_2 = TCanvas('cYields_2', 'The Fit Canvas 2')

    filtrated_phi_1 = filtrated_phi.query("pt_cand < 2")

    grouped_phi_1 = filtrated_phi_1.groupby(["run_number", "ev_id"])
    phi_vec_1     = grouped_phi_1["phi_cand"]

    d_phi_dist_1 = np.abs(phi_vec_1.max() - phi_vec_1.min()) #delta phi destribution as difference between max and min phi in the group

    filtrated_phi_2 = filtrated_phi.query("pt_cand > 2")
    filtrated_phi_2 = filtrated_phi_2.query("pt_cand < 3")

    grouped_phi_2 = filtrated_phi_2.groupby(["run_number", "ev_id"])
    phi_vec_2     = grouped_phi_2["phi_cand"]

    d_phi_dist_2 = np.abs(phi_vec_2.max() - phi_vec_2.min()) #delta phi destribution as difference between max and min phi in the group

    filtrated_phi_3 = filtrated_phi.query("pt_cand < 4")
    filtrated_phi_3 = filtrated_phi_3.query("pt_cand > 3")

    grouped_phi_3 = filtrated_phi_3.groupby(["run_number", "ev_id"])
    phi_vec_3     = grouped_phi_3["phi_cand"]

    d_phi_dist_3 = np.abs(phi_vec_3.max() - phi_vec_3.min()) #delta phi destribution as difference between max and min phi in the group

    filtrated_phi_4 = filtrated_phi.query("pt_cand < 5")
    filtrated_phi_4 = filtrated_phi_4.query("pt_cand > 4")

    grouped_phi_4 = filtrated_phi_4.groupby(["run_number", "ev_id"])
    phi_vec_4     = grouped_phi_4["phi_cand"]

    d_phi_dist_4 = np.abs(phi_vec_4.max() - phi_vec_4.min()) #delta phi destribution as difference between max and min phi in the group

    filtrated_phi_5 = filtrated_phi.query("pt_cand < 6")
    filtrated_phi_5 = filtrated_phi_5.query("pt_cand > 5")

    grouped_phi_5 = filtrated_phi_5.groupby(["run_number", "ev_id"])
    phi_vec_5     = grouped_phi_5["phi_cand"]

    d_phi_dist_5 = np.abs(phi_vec_5.max() - phi_vec_5.min()) #delta phi destribution as difference between max and min phi in the group

    h_d_phi_cand_1 = TH1F("delta phi cand, pt range:[1-2]" , "Normalized plot", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_1, d_phi_dist_1)
    h_d_phi_cand_1.Scale(1/ h_d_phi_cand_1.Integral())
    h_d_phi_cand_2 = TH1F("delta phi cand, pt range:[2-3]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_2, d_phi_dist_2)
    h_d_phi_cand_2.Scale(1/ h_d_phi_cand_2.Integral())
    h_d_phi_cand_3 = TH1F("delta phi cand, pt range:[3-4]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_3, d_phi_dist_3)
    h_d_phi_cand_3.Scale(1/ h_d_phi_cand_3.Integral())
    h_d_phi_cand_4 = TH1F("delta phi cand, pt range:[4-5]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_4, d_phi_dist_4)
    h_d_phi_cand_4.Scale(1/ h_d_phi_cand_4.Integral())
    h_d_phi_cand_5 = TH1F("delta phi cand, pt range:[5-6]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_5, d_phi_dist_5)
    h_d_phi_cand_5.Scale(1/ h_d_phi_cand_5.Integral())
    cYields_2.SetLogy(True)
    h_d_phi_cand_1.SetStats(0)
    h_d_phi_cand_1.SetLineColor(kBlack)
    h_d_phi_cand_1.Draw()
    h_d_phi_cand_2.SetStats(0)
    h_d_phi_cand_2.SetLineColor(kRed)
    h_d_phi_cand_2.Draw("same")
    h_d_phi_cand_3.SetStats(0)
    h_d_phi_cand_3.SetLineColor(kBlue)
    h_d_phi_cand_3.Draw("same")
    h_d_phi_cand_4.SetStats(0)
    h_d_phi_cand_4.SetLineColor(kGreen)
    h_d_phi_cand_4.Draw("same")
    h_d_phi_cand_5.SetStats(0)
    h_d_phi_cand_5.SetLineColor(kMagenta)
    h_d_phi_cand_5.Draw("same")
    leg = TLegend(0.45, 0.7, 0.95, 0.87)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(h_d_phi_cand_1, h_d_phi_cand_1.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_2, h_d_phi_cand_2.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_3, h_d_phi_cand_3.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_4, h_d_phi_cand_4.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_5, h_d_phi_cand_5.GetName(),"L")
    leg.Draw("same")
    cYields_2.SaveAs("h_d_phi_cand_compare.png")

cYields.SetLogy(False)
h_first_cand_mass = TH1F("inv_mass of the first cand" , "", 200,
        inv_mass_max_vec.min(), inv_mass_max_vec.max())
fill_hist(h_first_cand_mass, inv_mass_max_vec)
h_second_cand_mass = TH1F("inv_mass of the second cand" , "", 200,
        inv_mass_min_vec.min(), inv_mass_min_vec.max())
fill_hist(h_second_cand_mass, inv_mass_min_vec)
h_first_cand_mass.SetLineColor(kRed)
h_second_cand_mass.SetLineColor(kBlue)
h_first_cand_mass.Draw()
h_second_cand_mass.Draw("same")
cYields.SaveAs("h_inv_mass_cand.png")

h_first_cand_pt = TH1F("pt of the first cand" , "", 200,
        pt_max_vec.min(), pt_max_vec.max())
fill_hist(h_first_cand_pt, pt_max_vec)
h_second_cand_pt = TH1F("pt of the second cand" , "", 200,
        pt_min_vec.min(),pt_min_vec.max())
fill_hist(h_second_cand_pt, pt_min_vec)
h_first_cand_pt.SetLineColor(kRed)
h_second_cand_pt.SetLineColor(kBlue)
h_first_cand_pt.Draw()
h_second_cand_pt.Draw("same")
cYields.SaveAs("h_pt_cand_max_min.png")

h_first_cand_eta = TH1F("eta of the first cand" , "", 200,
        eta_max_vec.min(), eta_max_vec.max())
fill_hist(h_first_cand_eta, eta_max_vec)
h_second_cand_eta = TH1F("eta of the second cand" , "", 200,
        eta_min_vec.min(), eta_min_vec.max())
fill_hist(h_second_cand_eta, eta_min_vec)
h_first_cand_eta.SetLineColor(kRed)
h_second_cand_eta.SetLineColor(kBlue)
h_first_cand_eta.Draw()
h_second_cand_eta.Draw("same")
cYields.SaveAs("h_eta_cand_max_min.png")

#start = time.time()
#filtrated_eta = filtrated_eta.groupby(["run_number", "ev_id"])
#end1 = time.time()
#eta_vec     = filtrated_eta["eta_cand"]
#d_eta_dist = np.abs(eta_vec.max() - eta_vec.min())
#end2 = time.time()
#print("grouping eta", end1 - start, "sec")
#print("calc dist", end2 - end1, "sec")
#h_d_eta_cand = TH1F("delta eta cand" , "", 200, d_eta_dist.min(),
#        d_eta_dist.max())
#fill_hist(h_d_eta_cand, d_eta_dist)
#cYields.SetLogy(True)
#h_d_eta_cand.Draw()
#cYields.SaveAs("h_d_eta_cand.png")

d_eta_phi_dist = np.abs(eta_max_vec.values - eta_min_vec.values)

h_d_eta_cand = TH1F("delta eta cand" , "", 200, d_eta_phi_dist.min(),
        d_eta_phi_dist.max())
fill_hist(h_d_eta_cand, d_eta_phi_dist)
h_d_eta_cand.Draw()
cYields.SaveAs("h_d_eta_cand.png")


cYields = TCanvas('cYields', 'The Fit Canvas')
h_eta_phi = TH2F("delta eta/delta phi" , "", 200,
        d_eta_phi_dist.min(), d_eta_phi_dist.max(), 200, d_phi_dist.min(), d_phi_dist.max())
eta_phi = np.column_stack((d_eta_phi_dist, d_phi_dist))
fill_hist(h_eta_phi, eta_phi)
h_eta_phi.Draw("BOX")
cYields.SaveAs("h_eta_phi.png")


