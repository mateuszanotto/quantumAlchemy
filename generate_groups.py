def generate_unique_porphyrin_structures(max_charge):
    """
    Generates all unique substitution structures for a copper-porphyrin (D₄h symmetry)
    by partitioning carbons into symmetry-equivalent groups and avoiding duplicates.
    Returns a list of tuples representing substitution patterns (B = -1, N = +1, C = 0)
    where the total charge is between -max_charge and max_charge.
    """
    # Define symmetry groups (meso, beta-adjacent, beta-opposite)
    groups = [
        {
            "name": "beta_adjacent",
            "indices": [0, 5, 10, 15],  # 4 beta-adjacent positions
            "max_sub": 4
        },
        {
            "name": "alpha",
            "indices": [1, 6, 11, 16],  # 4 meso positions (hypothetical indices)
            "max_sub": 4
        },
        {
            "name": "alpha2",
            "indices": [3, 8, 13, 18],  # 4 meso positions (hypothetical indices)
            "max_sub": 4
        },
        {
            "name": "meso",
            "indices": [2, 7, 12, 17],  # 4 meso positions (hypothetical indices)
            "max_sub": 4
        },      
        {
            "name": "beta_opposite",
            "indices": [4, 9, 14, 19],  # 4 beta-opposite positions
            "max_sub": 4
        }
    ]

    # Precompute substitution patterns for a single group (canonical order)
    def _generate_group_patterns(group_indices, max_substitutions):
        patterns = []
        for k in range(0, max_substitutions + 1):  # Total substitutions in group
            for num_boron in range(0, k + 1):  # Number of boron substitutions
                # Calculate charge contribution for this group: (N - B) = (k - num_boron) - num_boron
                charge_contribution = (k - num_boron) - num_boron
                # Create substitution vector for this group
                sub_pattern = ([-1] * num_boron) + ([1] * (k - num_boron)) + [0] * (len(group_indices) - k)
                # Map substitutions to actual indices (canonical order)
                indexed_pattern = [(idx, sub) for idx, sub in zip(group_indices, sub_pattern)]
                patterns.append((k, charge_contribution, indexed_pattern))
        return patterns

    # Generate patterns for all groups
    beta_adj_patterns = _generate_group_patterns(groups[0]["indices"], groups[0]["max_sub"])
    alpha_patterns = _generate_group_patterns(groups[1]["indices"], groups[1]["max_sub"])
    alpha2_patterns = _generate_group_patterns(groups[2]["indices"], groups[2]["max_sub"])
    meso_patterns = _generate_group_patterns(groups[3]["indices"], groups[3]["max_sub"])
    beta_opp_patterns = _generate_group_patterns(groups[4]["indices"], groups[4]["max_sub"])

    # Combine patterns across groups and filter by total charge
    unique_structures = []
    for (k_beta_adj, charge_beta_adj, beta_adj_subs) in beta_adj_patterns:
        for (k_alpha, charge_alpha, alpha_subs) in alpha_patterns:
            for (k_meso, charge_meso, meso_subs) in meso_patterns:
                for (k_alpha2, charge_alpha2, alpha2_subs) in alpha2_patterns:
                    for (k_beta_opp, charge_beta_opp, beta_opp_subs) in beta_opp_patterns:
                        total_charge = charge_beta_adj + charge_alpha + charge_meso + charge_alpha2 + charge_beta_opp
                        if -max_charge <= total_charge <= max_charge:
                            # Merge substitutions into a single structure
                            structure = [0] * 20
                            for idx, sub in meso_subs + beta_adj_subs + beta_opp_subs + alpha_subs + alpha2_subs:
                                structure[idx] = sub
                            unique_structures.append(tuple(structure))

    return unique_structures

