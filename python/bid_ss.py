import config
from ss_mdp import SS_MDP
from mcpc import Mcpc
from lin_bid import Lin_Bid
from rlb_dp_i import RLB_DP_I
from rlb_dp_smooth import RLB_DP_SMOOTH
from utility import *
import os
import sys

obj_type = "clk"
clk_vp = 1
# N = 1000
gamma = 1

src = "ipinyou"

MODEL = int(sys.argv[1])
CAMP = int(sys.argv[2])
alpha = float(sys.argv[3])
beta = float(sys.argv[4])
c0 = float(sys.argv[5])
N = int(sys.argv[6])

if src == "ipinyou":
    camps = config.ipinyou_camps
    data_path = config.ipinyouPath
    full_data_path = config.full_data_path
    max_market_price = config.ipinyou_max_market_price
elif src == "vlion":
    camps = config.vlion_camps
    data_path = config.vlionPath
    max_market_price = config.vlion_max_market_price
elif src == "yoyi":
    camps = config.yoyi_camps
    data_path = config.yoyiPath
    max_market_price = config.yoyi_max_market_price

camp = camps[CAMP]
# for camp in camps[:]:
log_in = open(
    config.projectPath + "bid-performance/full_eval_{}_N={}_c0={}_obj={}_clkvp={}_model={}_camp={}_alpha={}_beta={}.txt".format(
        src, N,
        c0,
        obj_type,
        clk_vp,
        MODEL,
        camp,
        alpha,
        beta),
    "w")
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}\t" \
    .format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
log_in.write(log + "\n")

camp_info = config.get_camp_info(camp, src)
auction_in = open(data_path + camp + "/test.theta.txt", "r")
opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))
B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)

m_pdf = calc_m_pdf(camp_info["price_counter_train"])
rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
                                                          data_path + camp + "/bid-model/v_nb_N={}_c0={}.txt".format(N,
                                                                                                                     c0))
# rlb_dp_i.Vnb2Dnb(data_path + camp + "/bid-model/v_nb_N={}_c0={}.txt".format(N, c0),
#                  data_path + camp + "/bid-model/rlb_dnb_N={}_c0={}.txt".format(N, c0))

rlb_dp_smooth = RLB_DP_SMOOTH(camp_info, opt_obj, gamma, m_pdf, MODEL, alpha, beta)
rlb_dp_smooth.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf,
                                                               data_path + camp + "/bid-model/smooth_v_nb_N={}_model={}_alpha={}_beta={}_c0={}.txt".format(
                                                                   N, MODEL, alpha, beta, c0))

# SS-MDP
auction_in = open(full_data_path + camp + "/test.theta.txt", "r")
ss_mdp = SS_MDP(camp_info)
setting = "{}, camp={}, algo={}, N={}, c0={}" \
    .format(src, camp, "ss_mdp", N, c0)
bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

model_path = data_path + camp + "/bid-model/v_nb_N={}_c0={}.txt".format(N, c0)

ss_mdp.load_value_function(N, B, model_path)
(auction, imp, clk, cost) = ss_mdp.run(auction_in, bid_log_path, N, c0,
                                       max_market_price, delimiter=" ", save_log=False)

win_rate = imp / auction * 100
cpm = (cost / 1000) / imp * 1000
ecpc = (cost / 1000) / clk
obj = opt_obj.get_obj(imp, clk, cost)
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
    .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
print(log)
log_in.write(log + "\n")

# Mcpc
auction_in = open(full_data_path + camp + "/test.theta.txt", "r")
mcpc = Mcpc(camp_info)
setting = "{}, camp={}, algo={}, N={}, c0={}" \
    .format(src, camp, "mcpc", N, c0)
bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
(auction, imp, clk, cost) = mcpc.run(auction_in, bid_log_path, N, c0,
                                     max_market_price, delimiter=" ", save_log=False)
win_rate = imp / auction * 100
cpm = (cost / 1000) / imp * 1000
ecpc = (cost / 1000) / clk
obj = opt_obj.get_obj(imp, clk, cost)
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
    .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
print(log)
log_in.write(log + "\n")

# Lin-Bid
auction_in = open(full_data_path + camp + "/test.theta.txt", "r")
lin_bid = Lin_Bid(camp_info)
setting = "{}, camp={}, algo={}, N={}, c0={}" \
    .format(src, camp, "lin_bid", N, c0)
bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
model_path = data_path + camp + "/bid-model/{}_{}_{}_{}_{}.pickle".format("lin-bid", N, c0, obj_type, opt_obj.clk_v)
valid_path = data_path + camp + "/train.theta.txt"
lin_bid.parameter_tune(opt_obj, valid_path, model_path, N, c0, max_market_price,
                       max_market_price, delimiter=" ", load=True)
(auction, imp, clk, cost) = lin_bid.run(auction_in, bid_log_path, N, c0,
                                        max_market_price, delimiter=" ", save_log=False)

win_rate = imp / auction * 100
cpm = (cost / 1000) / imp * 1000
ecpc = (cost / 1000) / clk
obj = opt_obj.get_obj(imp, clk, cost)
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
    .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
print(log)
log_in.write(log + "\n")

# RLB
auction_in = open(full_data_path + camp + "/test.theta.txt", "r")
rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
setting = "{}, camp={}, algo={}, N={}, c0={}" \
    .format(src, camp, "rlb", N, c0)
bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

model_path = data_path + camp + "/bid-model/v_nb_N={}_c0={}.txt".format(N, c0)
rlb_dp_i.load_value_function(N, B, model_path)

(auction, imp, clk, cost) = rlb_dp_i.run(auction_in, bid_log_path, N, c0,
                                         max_market_price, delimiter=" ", save_log=False)

win_rate = imp / auction * 100
cpm = (cost / 1000) / imp * 1000
ecpc = (cost / 1000) / clk
clk_per_imp = clk / imp
obj = opt_obj.get_obj(imp, clk, cost)
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}\t" \
    .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
print(log)
log_in.write(log + "\n")

# RLB SMOOTH
auction_in = open(full_data_path + camp + "/test.theta.txt", "r")
rlb_dp_smooth = RLB_DP_SMOOTH(camp_info, opt_obj, gamma, m_pdf, MODEL, alpha, beta)
setting = "{}, camp={}, algo={}, N={}, c0={}" \
    .format(src, camp, "rlb_smooth", N, c0)
bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

model_path = data_path + camp + "/bid-model/smooth_v_nb_N={}_model={}_alpha={}_beta={}_c0={}.txt".format(N,
                                                                                                         MODEL, alpha,
                                                                                                         beta, c0)
rlb_dp_smooth.load_value_function(N, B, model_path)

(auction, imp, clk, cost) = rlb_dp_smooth.run(auction_in, bid_log_path, N, c0,
                                              max_market_price, delimiter=" ", save_log=False)

win_rate = imp / auction * 100
cpm = (cost / 1000) / imp * 1000
ecpc = (cost / 1000) / clk
obj = opt_obj.get_obj(imp, clk, cost)
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}\t" \
    .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
print(log)
log_in.write(log + "\n")

log_in.flush()
log_in.close()
