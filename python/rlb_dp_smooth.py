from utility import *
import config
import numpy as np


class RLB_DP_SMOOTH:
    up_precision = 1e-10
    zero_precision = 1e-12

    def __init__(self, camp_info, opt_obj, gamma, m_pdf, MODEL, alpha, beta):
        self.cpm = camp_info["cost_train"] / camp_info["imp_train"]
        self.theta_avg = camp_info["clk_train"] / camp_info["imp_train"]
        self.opt_obj = opt_obj
        self.gamma = gamma
        self.v1 = self.opt_obj.v1
        self.v0 = self.opt_obj.v0
        self.m_pdf = m_pdf
        self.MODEL = MODEL
        self.alpha = alpha
        self.beta = beta
        self.V = []
        self.D = []

    def penalty(self, n, b, N, B):
        if self.MODEL == 0:
            return (b - B * (n - 1) / N) ** 2
        elif self.MODEL == 1:
            return (self.beta ** (N - n)) * ((b - B * (n - 1) / N) ** 2)
        elif self.MODEL == 2:
            return abs(b - B * (n - 1) / N)

    def calc_optimal_value_function_with_approximation_i(self, N, B, max_bid, m_pdf, save_path):
        V_out = open(save_path, "w")
        V = np.zeros(B + 1)
        nV = np.zeros(B + 1)
        for n in range(1, N):
            for b in range(0, B):
                V_out.write("{}\t".format(V[b]))
            V_out.write("{}\n".format(V[B]))

            for b in range(1, B + 1):
                dp = np.zeros(min(max_bid, b) + 1)
                dp[0] = m_pdf[0] * (self.v1 * self.theta_avg)
                for delta in range(1, min(max_bid, b) + 1):
                    dp[delta] = dp[delta - 1] + m_pdf[delta] * (
                            self.v1 * self.theta_avg + self.gamma * (V[b - delta] - V[b]) + self.alpha * (
                            self.penalty(n, b, N, B) - self.penalty(n, b - delta, N, B)) - self.v0 * delta)
                nV[b] = np.max(dp) + self.gamma * V[b] - self.alpha * self.penalty(n, b, N, B)

            V = nV[:]

        for b in range(0, B):
            V_out.write("{0}\t".format(V[b]))
        V_out.write("{0}\n".format(V[B]))
        V_out.flush()
        V_out.close()

    def load_value_function(self, N, B, model_path):
        self.V = [[0 for i in range(B + 1)] for j in range(N)]
        with open(model_path, "r") as fin:
            n = 0
            for line in fin:
                line = line[:len(line) - 1].split("\t")
                for b in range(B + 1):
                    self.V[n][b] = float(line[b])
                n += 1
                if n >= N:
                    break

    def bid(self, n, b, theta, max_bid, N, B):
        dp = np.zeros(min(b, max_bid) + 1)
        dp[0] = self.m_pdf[0] * (self.v1 * theta)
        for delta in range(1, min(b, max_bid) + 1):
            dp[delta] = dp[delta - 1] + self.m_pdf[delta] * (
                    self.v1 * theta + self.gamma * (self.V[n - 1][b - delta] - self.V[n - 1][b]) + self.alpha * (
                    self.penalty(n, b, N, B) - self.penalty(n, b - delta, N, B)) - self.v0 * delta)
        return np.argmax(dp)

    def run(self, auction_in, bid_log_path, N, c0, max_bid, input_type="file reader", delimiter=" ", save_log=False):
        auction = 0
        imp = 0
        clk = 0
        cost = 0

        if save_log:
            log_in = open(bid_log_path, "w")
        B = int(self.cpm * c0 * N)

        episode = 1
        n = N
        b = B
        b_arr = []
        for line in auction_in:
            if input_type == "file reader":
                line = line[:len(line) - 1].split(delimiter)
                click = int(line[0])
                price = int(line[1])
                theta = float(line[2])
            else:
                (click, price, theta) = line
            a = self.bid(n, b, theta, max_bid, N, B)
            a = min(int(a), min(b, max_bid))

            log = getTime() + "\t{}\t{}_{}\t{}_{}_{}\t{}_{}\t".format(
                episode, b, n, a, price, click, clk, imp)
            if save_log:
                log_in.write(log + "\n")

            if a >= price:
                imp += 1
                if click == 1:
                    clk += 1
                b -= price
                cost += price
            n -= 1
            auction += 1
            # b_arr.append(b)
            if n == 0:
                episode += 1
                n = N
                b = B
        # np.savetxt("/home/mohammad.afzali96.student/RTBidding/rlb_smooth.txt", b_arr)
        if save_log:
            log_in.flush()
            log_in.close()

        return auction, imp, clk, cost

    def Vnb2Dnb(self, v_path, d_path):
        with open(v_path, "r") as fin:
            with open(d_path, "w") as fout:
                for line in fin:
                    line = line[:len(line) - 1].split("\t")
                    nl = ""
                    for b in range(len(line) - 1):
                        d = float(line[b + 1]) - float(line[b])
                        if abs(d) < RLB_DP_SMOOTH.zero_precision:
                            d = 0
                        if b == len(line) - 2:
                            nl += "{}\n".format(d)
                        else:
                            nl += "{}\t".format(d)
                    fout.write(nl)
