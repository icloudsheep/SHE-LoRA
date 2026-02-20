#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   negotiation.py
@Time    :   2024/12/28 20:05:26
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Negotiate common encryption range
'''

from flwr.server.client_proxy import ClientProxy
# from flwr.server.client_manager import ClientManager
import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from skopt import gp_minimize
    from skopt.space import Real
    _HAS_SKOPT = True
except ImportError:
    _HAS_SKOPT = False


def embed_client_budget(client: ClientProxy, rank: int, budget: int):
    '''
    Embed privacy budget into client properties
    '''
    client.properties['lora_rank'] = rank
    client.properties['he_budget'] = budget
    return client


def get_common_sensitive(results: dict[int,dict])->dict:
    '''
    results: dictionary, key is client_id, value is scores dictionary [str,list[dict]] 
    '''
    common_dict_list :dict[str,dict[int,int]] = {}
    sensitive_dict_list :dict[str,dict[int,int]] = {}

    for client_scores in results.values():
        for layer, scores in client_scores.items():
            '''
            layer: <str> // layer_0_self_attn.q_proj
            scores: list[dict[str,float]] // [{'line': 2031, 'score': 4.425048351287842}, {'line': 718, 'score': 2.6232924461364746}]
            '''

            current_layer_lines = common_dict_list[layer] if layer in sensitive_dict_list else {}
            local_sensitive_dict = sensitive_dict_list[layer] if layer in sensitive_dict_list else {}

            for line in scores:
                if current_layer_lines is None:
                    current_layer_lines[line['line']] = 1
                elif line['line'] not in current_layer_lines.keys():
                    current_layer_lines[line['line']] = 1
                else:
                    current_layer_lines[line['line']] += 1
                
                if local_sensitive_dict is None:
                    local_sensitive_dict[line['line']] = line['score']
                elif line['line'] not in local_sensitive_dict.keys():
                    local_sensitive_dict[line['line']] = line['score']
                else:
                    local_sensitive_dict[line['line']] = max(local_sensitive_dict[line['line']], line['score'])
            common_dict_list[layer] = current_layer_lines
            sensitive_dict_list[layer] = local_sensitive_dict

    for layer in common_dict_list:
        common_dict_list[layer] = dict(sorted(common_dict_list[layer].items(), key=lambda item: item[1], reverse=True))

    for layer in sensitive_dict_list:
        sensitive_dict_list[layer] = dict(sorted(sensitive_dict_list[layer].items(), key=lambda item: item[1], reverse=True))

    return common_dict_list, sensitive_dict_list


def _build_clients_list_for_budget(
    scores: dict,
    layer: str,
    budget: int,
    cids_with_budget: List[int],
) -> List[int]:
    """
    Build the Clients list for one layer and budget: unique columns from budget-k clients,
    ranked by minimum sensitivity (desc). Paper: "ranks unique columns from budget-k
    clients by their minimum sensitivity".
    Returns list of line ids (column indices) in order of min_sensitivity descending.
    """
    # line -> list of (score from each client that has this line in top-k)
    line_scores_by_client: Dict[int, List[float]] = defaultdict(list)
    for cid in cids_with_budget:
        if layer not in scores.get(cid, {}):
            continue
        sorted_items = sorted(
            scores[cid][layer],
            key=lambda x: x["score"],
            reverse=True,
        )
        top_k_lines = sorted_items[:budget]
        for item in top_k_lines:
            line, score = item["line"], item["score"]
            line_scores_by_client[line].append(score)
    # min sensitivity per line (only lines that appear in at least one client's top-k)
    line_min_sens = [
        (line, min(scores_list))
        for line, scores_list in line_scores_by_client.items()
    ]
    line_min_sens.sort(key=lambda x: x[1], reverse=True)
    return [line for line, _ in line_min_sens]


def _compute_coverage_risk(
    scores: dict,
    enclines: Dict[str, List[int]],
    client_budgets: Dict[int, int],
) -> Tuple[float, float, float]:
    """
    Compute min-Coverage, max-Risk, and composite score (Eq. 14).
    Coverage_i = |Res ∩ G_i| / |G_i|, Risk_i = sum(S_j for j in G_i \\ Res) / sum(S_j for j in G_i).
    score(Res) = min_i Coverage_i - max_i Risk_i (maximize).
    Returns (min_coverage, max_risk, score).
    """
    coverage_list = []
    risk_list = []
    for cid, budget in client_budgets.items():
        if cid not in scores:
            continue
        gi_sets = {}
        gi_sensitivity_sum = {}
        for layer, items in scores[cid].items():
            sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
            top_k = sorted_items[:budget]
            gi_sets[layer] = set(item["line"] for item in top_k)
            gi_sensitivity_sum[layer] = sum(item["score"] for item in top_k)
        # Aggregate over layers: treat as union for this client (or average per-layer)
        # Paper defines per-client: so we use per-layer then take min coverage and max risk over layers for this client, then min over clients for coverage and max over clients for risk.
        client_coverage_min = 1.0
        client_risk_max = 0.0
        for layer in gi_sets:
            res_layer = set(enclines.get(layer, []))
            gi = gi_sets[layer]
            if not gi:
                continue
            covered = len(res_layer & gi) / len(gi)
            client_coverage_min = min(client_coverage_min, covered)
            total_sens = gi_sensitivity_sum[layer]
            if total_sens <= 0:
                continue
            # unencrypted sensitivity for this client's G_i
            unenc_sens = 0.0
            for item in scores[cid][layer]:
                if item["line"] in gi and item["line"] not in res_layer:
                    unenc_sens += item["score"]
            risk_i = unenc_sens / total_sens
            client_risk_max = max(client_risk_max, risk_i)
        coverage_list.append(client_coverage_min)
        risk_list.append(client_risk_max)
    if not coverage_list:
        return 0.0, 1.0, -1.0
    min_coverage = min(coverage_list)
    max_risk = max(risk_list)
    score_val = min_coverage - max_risk
    return min_coverage, max_risk, score_val


def _select_res_with_abc(
    lam: int,
    res_prev: set,
    clients_list: List[int],
    common_list: List[int],
    sensitivity_list: List[int],
    a: float,
    b: float,
    c: float,
) -> set:
    """
    Select columns by (a, b, c): ⌊aλ⌋ from Clients, ⌊bλ⌋ from Common,
    λ - ⌊aλ⌋ - ⌊bλ⌋ from Sensitivity, no duplicates. Returns res_prev ∪ new selections.
    """
    res = set(res_prev)
    na = max(0, int(a * lam))
    nb = max(0, int(b * lam))
    nc = max(0, lam - na - nb)
    # P from Clients
    for line in clients_list:
        if len(res) - len(res_prev) >= na:
            break
        if line not in res:
            res.add(line)
    # C from Common
    for line in common_list:
        if len(res) - len(res_prev) >= na + nb:
            break
        if line not in res:
            res.add(line)
    # S from Sensitivity
    for line in sensitivity_list:
        if len(res) - len(res_prev) >= na + nb + nc:
            break
        if line not in res:
            res.add(line)
    return res


def bayesian_optimize_abc(
    scores: dict,
    common_dict_list: dict,
    sensitive_dict_list: dict,
    clients: List[ClientProxy],
    budget: int,
    previous_budget: int,
    res_prev_per_layer: Dict[str, List[int]],
    n_iter: int = 50,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Bayesian optimization for (a, b, c) to maximize score(Res) = min-Coverage - max-Risk
    (Eq. 14). Search space: (a, b) in [0,1]^2 with a + b <= 1, c = 1 - a - b.
    Returns (a_star, b_star, c_star).
    """
    lam = budget - previous_budget
    if lam <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

    cids_with_budget = [int(c.cid) for c in clients if c.properties.get("he_budget") == budget]
    client_budgets = {int(c.cid): c.properties["he_budget"] for c in clients}

    # Build per-layer lists (read-only views for this tier)
    layers = list(common_dict_list.keys())
    clients_list_per_layer = {}
    common_list_per_layer = {}
    sensitivity_list_per_layer = {}
    for layer in layers:
        clients_list_per_layer[layer] = _build_clients_list_for_budget(
            scores, layer, budget, cids_with_budget
        )
        common_list_per_layer[layer] = list(common_dict_list[layer].keys())
        sensitivity_list_per_layer[layer] = list(sensitive_dict_list[layer].keys())

    def objective(x: List[float]) -> float:
        a_val, b_val = float(x[0]), float(x[1])
        c_val = 1.0 - a_val - b_val
        if c_val < 0:
            return 1.0  # invalid, maximize -> return bad score
        # Build hypothetical enclines for this tier: res_prev + new selection per layer
        enclines = {}
        for layer in layers:
            res_prev = set(res_prev_per_layer.get(layer, []))
            res_new = _select_res_with_abc(
                lam,
                res_prev,
                clients_list_per_layer[layer],
                common_list_per_layer[layer],
                sensitivity_list_per_layer[layer],
                a_val,
                b_val,
                c_val,
            )
            enclines[layer] = list(res_new)
        min_cov, max_r, score_val = _compute_coverage_risk(scores, enclines, client_budgets)
        return -score_val  # minimize negative score = maximize score

    if _HAS_SKOPT:
        space = [Real(0.0, 1.0), Real(0.0, 1.0)]
        res = gp_minimize(
            objective,
            space,
            n_calls=n_iter,
            random_state=random_state,
            n_initial_points=min(10, n_iter),
        )
        a_star, b_star = res.x[0], res.x[1]
        c_star = 1.0 - a_star - b_star
        if c_star < 0:
            c_star = 0.0
            s = a_star + b_star
            if s > 0:
                a_star, b_star = a_star / s, b_star / s
        return (float(a_star), float(b_star), float(c_star))
    else:
        # Fallback: random search on simplex
        best_score = -2.0
        best_abc = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        rng = random.Random(random_state)
        for _ in range(n_iter):
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            a_val, b_val = u, v
            c_val = 1.0 - a_val - b_val
            score_val = -objective([a_val, b_val])
            if score_val > best_score:
                best_score = score_val
                best_abc = (a_val, b_val, c_val)
        return best_abc


def get_nego_enclines_bayesian(
    clients: List[ClientProxy],
    scores: dict,
    n_iter: int = 50,
    random_state: Optional[int] = None,
) -> dict:
    """
    Negotiate enc_lines using Bayesian-optimized (a, b, c) per budget tier.
    Maximizes min-Coverage and minimizes max-Risk (Eq. 14). Uses bayesian_optimize_abc
    for each budget level then selects columns by ⌊aλ⌋ from Clients, ⌊bλ⌋ from Common,
    and the rest from Sensitivity.
    """
    import copy
    # Deep copy so we don't mutate caller's scores
    scores_work = {}
    for cid, client_scores in scores.items():
        scores_work[cid] = {
            layer: [s.copy() for s in layer_scores]
            for layer, layer_scores in client_scores.items()
        }
    common_dict_list, sensitive_dict_list = get_common_sensitive(scores_work)
    budget_list = sorted(set(c.properties["he_budget"] for c in clients))
    enclines = {}
    previous_budget = 0
    for budget in budget_list:
        lam = budget - previous_budget
        res_prev_per_layer = {layer: list(enclines.get(layer, [])) for layer in common_dict_list}
        a_star, b_star, c_star = bayesian_optimize_abc(
            scores_work,
            copy.deepcopy(common_dict_list),
            copy.deepcopy(sensitive_dict_list),
            clients,
            budget,
            previous_budget,
            res_prev_per_layer,
            n_iter=n_iter,
            random_state=random_state,
        )
        cids_budget = [c for c in clients if c.properties.get("he_budget") == budget]
        selected_lines = {layer: list(enclines.get(layer, [])) for layer in common_dict_list}
        for layer in common_dict_list:
            common_list = list(common_dict_list[layer].keys())
            sensitivity_list = list(sensitive_dict_list[layer].keys())
            clients_list = _build_clients_list_for_budget(
                scores_work, layer, budget, [int(c.cid) for c in cids_budget]
            )
            res_prev = set(selected_lines[layer])
            res_new = _select_res_with_abc(
                lam, res_prev, clients_list, common_list, sensitivity_list,
                a_star, b_star, c_star,
            )
            added = res_new - res_prev
            selected_lines[layer] = list(res_new)
            for line in added:
                common_dict_list[layer].pop(line, None)
                sensitive_dict_list[layer].pop(line, None)
            for c in clients:
                cid = int(c.cid)
                if cid in scores_work and layer in scores_work[cid]:
                    scores_work[cid][layer] = [
                        item for item in scores_work[cid][layer]
                        if item["line"] not in added
                    ]
        enclines = selected_lines
        previous_budget = budget
    return enclines
