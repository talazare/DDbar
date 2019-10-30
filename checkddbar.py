import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ROOT import TH1F, TCanvas
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
import lz4.frame

#debug = True
debug = False

dfreco1 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco2_4_0.75.pkl.lz4", "rb"))
dfreco2 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))
dfreco3 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco6_8_0.65.pkl.lz4", "rb"))
#dfreco4 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco8_24_0.65.pkl.lz4", "rb"))
frames = [dfreco1, dfreco2, dfreco3]
dfreco = pd.concat(frames)
print("Data loaded")
print(dfreco.columns)
dfreco = dfreco.query("y_test_probxgboost>0.8")

if(debug):
    print("Debug mode: removing events")
    dfreco = dfreco[:20000]
print("Size of data", dfreco.shape)

cYields = TCanvas('cYields', 'The Fit Canvas')

h_invmass = TH1F("invariant mass" , "", 200, 1.64, 2.1)
fill_hist(h_invmass, dfreco.inv_mass)
h_invmass.Draw()
cYields.SaveAs("h_invmass.pdf")

h_pt_prong0 = TH1F("pt prong_0" , "", 200, 0.8, 6.)
fill_hist(h_pt_prong0, dfreco.pt_prong0)
h_pt_prong0.Draw()
cYields.SaveAs("h_pt_prong0.pdf")


h_pt_prong1 = TH1F("pt prong_1" , "", 200, 0.8, 6.)
fill_hist(h_pt_prong1, dfreco.pt_prong1)
h_pt_prong1.Draw()
cYields.SaveAs("h_pt_prong1.pdf")


h_eta_prong0 = TH1F("eta prong_0" , "", 200, 0., 0.8)
fill_hist(h_eta_prong0, dfreco.eta_prong0)
h_eta_prong0.Draw()
cYields.SaveAs("h_eta_prong0.pdf")


h_eta_prong1 = TH1F("eta prong_1" , "", 200, 0., 0.8)
fill_hist(h_eta_prong1, dfreco.eta_prong1)
h_eta_prong1.Draw()
cYields.SaveAs("h_eta_prong1.pdf")


h_eta_cand = TH1F("eta cand" , "", 200, 0., 0.8)
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


grouped = dfreco.groupby(["run_number","ev_id"])
#.filter(lambda x: len(x) > 1).groupby(["run_number","ev_id"])
grouplen = pd.array([len(group) for name, group in grouped])
h_grouplen = TH1F("group_length" , "", 5, 1., 6.)
fill_hist(h_grouplen, grouplen)
h_grouplen.Draw()
cYields.SaveAs("h_grouplen.pdf")

filtrated_phi = grouped.filter(lambda x: x.phi_cand.max() - x.phi_cand.min() >
        0).groupby(["run_number", "ev_id"])
filtrated_eta = grouped.filter(lambda x: x.eta_cand.max() - x.eta_cand.min() >
        0).groupby(["run_number", "ev_id"])

print("Grouped and filtered")

phi_vec     = filtrated_phi["phi_cand"]
d_phi_dist = np.abs(phi_vec.max() - phi_vec.min())
#print(d_phi_dist)
h_d_phi_cand = TH1F("delta phi cand" , "", 200, 0., 5.)
fill_hist(h_d_phi_cand, d_phi_dist)
h_d_phi_cand.Draw()
cYields.SaveAs("h_d_phi_cand.pdf")

eta_vec     = filtrated_eta["eta_cand"]
d_eta_dist = np.abs(eta_vec.max() - eta_vec.min())
#print(d_eta_dist)
h_d_eta_cand = TH1F("delta eta cand" , "", 200, 0., 3.)
fill_hist(h_d_eta_cand, d_eta_dist)
h_d_eta_cand.Draw()
cYields.SaveAs("h_d_eta_cand.pdf")
