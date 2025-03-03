# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import math
from numba import jit
from tqdm import tqdm

from dataset import Dataset, TrainDataset, AdapTestDataset
from setting import params

# ---------------------
# 1. Load data như code gốc nhưng tối ưu hóa
# ---------------------


def load_data(train_file, metadata_file):
    triplets = pd.read_csv(
        train_file, encoding='utf-8').to_records(index=False)
    metadata = json.load(open(metadata_file, 'r'))
    train_data = AdapTestDataset(
        triplets, metadata['num_train_students'], metadata['num_questions'])

    data_j = []
    data_k = []
    data_y = []

    # Sử dụng list comprehension để tăng tốc
    for j, items in train_data.data.items():
        for k, y in items.items():
            data_j.append(j)
            data_k.append(k)
            data_y.append(y)

    # Chuyển đổi thành numpy array một lần
    data_j = np.array(data_j, dtype=np.int32)
    data_k = np.array(data_k, dtype=np.int32)
    data_y = np.array(data_y, dtype=np.int32)

    data_dict = {
        'J': metadata['num_train_students'],
        'K': metadata['num_questions'],
        'N': len(data_y),
        'j': data_j,
        'k': data_k,
        'y': data_y
    }

    return data_dict, metadata

# ---------------------
# 2. Định nghĩa hàm tính log-posterior với Numba để tăng tốc
# ---------------------


@jit(nopython=True)
def log_prior(mu_beta, sigma_beta, sigma_gamma, alpha, beta, gamma):
    """Tính log prior của các tham số"""
    # Kiểm tra các ràng buộc
    if sigma_beta <= 0 or sigma_gamma <= 0 or np.any(gamma <= 0):
        return -np.inf

    # mu_beta ~ Cauchy(0,5)
    log_prior_mu_beta = -np.log(5 * math.pi * (1 + (mu_beta/5)**2))

    # sigma_beta ~ Cauchy(0,5)
    log_prior_sigma_beta = -np.log(5 * math.pi * (1 + (sigma_beta/5)**2))

    # sigma_gamma ~ Cauchy(0,5)
    log_prior_sigma_gamma = -np.log(5 * math.pi * (1 + (sigma_gamma/5)**2))

    # alpha ~ N(0,1)
    log_prior_alpha = -0.5 * np.sum(alpha**2)

    # beta ~ N(0, sigma_beta)
    log_prior_beta = -0.5 * \
        np.sum((beta**2) / (sigma_beta**2)) - len(beta) * np.log(sigma_beta)

    # gamma ~ LogNormal(0, sigma_gamma)
    log_prior_gamma = -0.5 * np.sum((np.log(gamma)**2) / (sigma_gamma**2)) - np.sum(
        np.log(gamma)) - len(gamma)*np.log(sigma_gamma)

    return log_prior_mu_beta + log_prior_sigma_beta + log_prior_sigma_gamma + log_prior_alpha + log_prior_beta + log_prior_gamma


@jit(nopython=True)
def log_likelihood(alpha, beta, gamma, mu_beta, j_idx, k_idx, y):
    """Tính log likelihood"""
    # Sử dụng vectorization thay vì vòng lặp
    logits = gamma[k_idx] * (alpha[j_idx] - (beta[k_idx] + mu_beta))

    # Cách tính log likelihood an toàn
    pos_part = y * (-np.log1p(np.exp(-logits)))
    neg_part = (1 - y) * (-np.log1p(np.exp(logits)))

    return np.sum(pos_part + neg_part)


@jit(nopython=True)
def log_posterior(mu_beta, sigma_beta, sigma_gamma, alpha, beta, gamma, j_idx, k_idx, y):
    """Tính log-posterior = log(prior) + log(likelihood)"""
    lp = log_prior(mu_beta, sigma_beta, sigma_gamma, alpha, beta, gamma)

    if np.isinf(lp):
        return -np.inf

    ll = log_likelihood(alpha, beta, gamma, mu_beta, j_idx, k_idx, y)
    return lp + ll

# ---------------------
# 3. Viết lại hàm MCMC sampler với block updates và vectorization
# ---------------------


def mcmc_sampler(initial_params, data, num_iterations, proposal_scales, thinning=10, burn_in_pct=0.2):
    """
    MCMC sampler tối ưu:
    - Sử dụng block updates cho các vector thay vì cập nhật từng phần tử
    - Lưu mẫu sau thinning để giảm bộ nhớ và tương quan
    - Theo dõi tiến trình với tqdm
    """
    J = data['J']
    K = data['K']
    j_idx = data['j']
    k_idx = data['k']
    y = data['y']

    # Sao chép các tham số hiện tại
    current = {
        'mu_beta': initial_params['mu_beta'],
        'sigma_beta': initial_params['sigma_beta'],
        'sigma_gamma': initial_params['sigma_gamma'],
        'alpha': initial_params['alpha'].copy(),
        'beta': initial_params['beta'].copy(),
        'gamma': initial_params['gamma'].copy()
    }

    current_log_post = log_posterior(
        current['mu_beta'], current['sigma_beta'], current['sigma_gamma'],
        current['alpha'], current['beta'], current['gamma'],
        j_idx, k_idx, y
    )

    # Số lượng mẫu sau thinning
    num_samples = num_iterations // thinning

    # Lưu lại các mẫu sau thinning
    samples = {
        'mu_beta': np.zeros(num_samples),
        'sigma_beta': np.zeros(num_samples),
        'sigma_gamma': np.zeros(num_samples),
        'alpha': np.zeros((num_samples, J)),
        'beta': np.zeros((num_samples, K)),
        'gamma': np.zeros((num_samples, K))
    }

    # Khởi tạo biến đếm
    accept_count = {
        'mu_beta': 0, 'sigma_beta': 0, 'sigma_gamma': 0,
        'alpha': 0, 'beta': 0, 'gamma': 0
    }
    total_count = {
        'mu_beta': 0, 'sigma_beta': 0, 'sigma_gamma': 0,
        'alpha': 0, 'beta': 0, 'gamma': 0
    }

    # Sử dụng tqdm để hiển thị tiến trình
    for it in tqdm(range(num_iterations)):
        # --- Cập nhật mu_beta (scalar) ---
        total_count['mu_beta'] += 1
        prop_mu_beta = current['mu_beta'] + \
            proposal_scales['mu_beta'] * np.random.randn()
        log_post_prop = log_posterior(
            prop_mu_beta, current['sigma_beta'], current['sigma_gamma'],
            current['alpha'], current['beta'], current['gamma'],
            j_idx, k_idx, y
        )
        if np.log(np.random.rand()) < log_post_prop - current_log_post:
            current['mu_beta'] = prop_mu_beta
            current_log_post = log_post_prop
            accept_count['mu_beta'] += 1

        # --- Cập nhật sigma_beta (scalar, >0) ---
        total_count['sigma_beta'] += 1
        prop_sigma_beta = current['sigma_beta'] * \
            np.exp(proposal_scales['sigma_beta'] * np.random.randn())
        log_post_prop = log_posterior(
            current['mu_beta'], prop_sigma_beta, current['sigma_gamma'],
            current['alpha'], current['beta'], current['gamma'],
            j_idx, k_idx, y
        )
        # Thêm Jacobian cho biến đổi log-scale
        log_accept_ratio = log_post_prop - current_log_post + \
            np.log(prop_sigma_beta/current['sigma_beta'])
        if np.log(np.random.rand()) < log_accept_ratio:
            current['sigma_beta'] = prop_sigma_beta
            current_log_post = log_post_prop
            accept_count['sigma_beta'] += 1

        # --- Cập nhật sigma_gamma (scalar, >0) ---
        total_count['sigma_gamma'] += 1
        prop_sigma_gamma = current['sigma_gamma'] * \
            np.exp(proposal_scales['sigma_gamma'] * np.random.randn())
        log_post_prop = log_posterior(
            current['mu_beta'], current['sigma_beta'], prop_sigma_gamma,
            current['alpha'], current['beta'], current['gamma'],
            j_idx, k_idx, y
        )
        # Thêm Jacobian cho biến đổi log-scale
        log_accept_ratio = log_post_prop - current_log_post + \
            np.log(prop_sigma_gamma/current['sigma_gamma'])
        if np.log(np.random.rand()) < log_accept_ratio:
            current['sigma_gamma'] = prop_sigma_gamma
            current_log_post = log_post_prop
            accept_count['sigma_gamma'] += 1

        # --- Cập nhật vector alpha theo block (8-10 phần tử mỗi lần) ---
        total_count['alpha'] += 1
        block_size = min(10, J)
        for start_idx in range(0, J, block_size):
            end_idx = min(start_idx + block_size, J)

            prop_alpha = current['alpha'].copy()
            prop_alpha[start_idx:end_idx] += proposal_scales['alpha'] * \
                np.random.randn(end_idx - start_idx)

            log_post_prop = log_posterior(
                current['mu_beta'], current['sigma_beta'], current['sigma_gamma'],
                prop_alpha, current['beta'], current['gamma'],
                j_idx, k_idx, y
            )

            if np.log(np.random.rand()) < log_post_prop - current_log_post:
                current['alpha'] = prop_alpha
                current_log_post = log_post_prop
                accept_count['alpha'] += 1

        # --- Cập nhật vector beta theo block (8-10 phần tử mỗi lần) ---
        total_count['beta'] += 1
        block_size = min(10, K)
        for start_idx in range(0, K, block_size):
            end_idx = min(start_idx + block_size, K)

            prop_beta = current['beta'].copy()
            prop_beta[start_idx:end_idx] += proposal_scales['beta'] * \
                np.random.randn(end_idx - start_idx)

            log_post_prop = log_posterior(
                current['mu_beta'], current['sigma_beta'], current['sigma_gamma'],
                current['alpha'], prop_beta, current['gamma'],
                j_idx, k_idx, y
            )

            if np.log(np.random.rand()) < log_post_prop - current_log_post:
                current['beta'] = prop_beta
                current_log_post = log_post_prop
                accept_count['beta'] += 1

        # --- Cập nhật vector gamma theo block, sử dụng log-normal proposal ---
        total_count['gamma'] += 1
        block_size = min(10, K)
        for start_idx in range(0, K, block_size):
            end_idx = min(start_idx + block_size, K)

            prop_gamma = current['gamma'].copy()
            # Sử dụng log-scale để đảm bảo gamma > 0
            log_gamma_current = np.log(prop_gamma[start_idx:end_idx])
            log_gamma_prop = log_gamma_current + \
                proposal_scales['gamma'] * np.random.randn(end_idx - start_idx)
            prop_gamma[start_idx:end_idx] = np.exp(log_gamma_prop)

            log_post_prop = log_posterior(
                current['mu_beta'], current['sigma_beta'], current['sigma_gamma'],
                current['alpha'], current['beta'], prop_gamma,
                j_idx, k_idx, y
            )

            # Thêm Jacobian cho biến đổi log-scale
            log_accept_ratio = log_post_prop - current_log_post + \
                np.sum(log_gamma_prop - log_gamma_current)

            if np.log(np.random.rand()) < log_accept_ratio:
                current['gamma'] = prop_gamma
                current_log_post = log_post_prop
                accept_count['gamma'] += 1

        # Lưu mẫu sau thinning
        if (it + 1) % thinning == 0:
            idx = (it + 1) // thinning - 1
            samples['mu_beta'][idx] = current['mu_beta']
            samples['sigma_beta'][idx] = current['sigma_beta']
            samples['sigma_gamma'][idx] = current['sigma_gamma']
            samples['alpha'][idx] = current['alpha'].copy()
            samples['beta'][idx] = current['beta'].copy()
            samples['gamma'][idx] = current['gamma'].copy()

    # Tính tỉ lệ chấp nhận
    acceptance_rates = {k: v / total_count[k] for k, v in accept_count.items()}

    return samples, acceptance_rates, int(burn_in_pct * num_samples)

# ---------------------
# 4. Hàm để tính các ước lượng sau burn-in
# ---------------------


def compute_posterior_estimates(samples, burn_in):
    """Tính các ước lượng sau burn-in"""
    estimates = {}
    for key, value in samples.items():
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                estimates[key] = np.mean(value[burn_in:], axis=0)
            else:
                estimates[key] = np.mean(value[burn_in:])

    return estimates

# ---------------------
# 5. Hàm chính để chạy toàn bộ quá trình
# ---------------------


def run_irt_mcmc(train_file, metadata_file, output_dir, num_iterations=10000, thinning=10):
    """Hàm chính để chạy toàn bộ quá trình"""
    print("Đang tải dữ liệu...")
    data_dict, metadata = load_data(train_file, metadata_file)

    J = data_dict['J']
    K = data_dict['K']

    print(
        f"Dữ liệu có {J} học sinh và {K} câu hỏi với {data_dict['N']} quan sát")

    # Khởi tạo các tham số
    initial_params = {
        'mu_beta': 0.0,
        'sigma_beta': 1.0,
        'sigma_gamma': 1.0,
        'alpha': np.zeros(J),
        'beta': np.zeros(K),
        'gamma': np.ones(K)
    }

    # Điều chỉnh proposal_scales phù hợp
    proposal_scales = {
        'mu_beta': 0.05,
        'sigma_beta': 0.05,
        'sigma_gamma': 0.05,
        'alpha': 0.1,
        'beta': 0.1,
        'gamma': 0.1
    }

    print("Bắt đầu chạy MCMC...")
    samples, acceptance_rates, burn_in = mcmc_sampler(
        initial_params, data_dict, num_iterations,
        proposal_scales, thinning=thinning
    )

    print("Tính toán các ước lượng...")
    estimates = compute_posterior_estimates(samples, burn_in)

    # Lưu tham số (theo code gốc)
    print("Lưu các ước lượng...")
    import os
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/beta.npy", estimates['beta'] + estimates['mu_beta'])
    np.save(f"{output_dir}/alpha.npy", estimates['gamma'])

    print("Tỉ lệ chấp nhận:")
    for key, rate in acceptance_rates.items():
        print(f"{key}: {rate:.4f}")

    return samples, estimates, acceptance_rates


# Nếu chạy trực tiếp
if __name__ == "__main__":
    train_file = '/mnt/c/Users/Admin/Desktop/code python/CCAT/CCAT/data/NIPS2020/train_triples.csv'
    metadata_file = '/mnt/c/Users/Admin/Desktop/code python/CCAT/CCAT/data/NIPS2020/metadata.json'
    output_dir = params.data_name

    # Sử dụng nhiều mẫu hơn và thinning để giảm tương quan
    samples, estimates, acceptance_rates = run_irt_mcmc(
        train_file, metadata_file, output_dir,
        num_iterations=100, thinning=20
    )  # giảm num iteration

    print("Hoàn thành quá trình MCMC!")
