from scipy import stats

def two_sample_ttest(sample1, sample2, alpha=0.05):
    """
    Performs a two-sample t-test on the given data samples.
    
    Parameters:
    - sample1, sample2: Lists or arrays of sample data.
    - alpha: Significance level (default is 0.05).
    
    Returns:
    - t_stat: Calculated t-statistic.
    - p_value: Two-tailed p-value.
    - decision: Whether to reject the null hypothesis.
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    
    if p_value < alpha:
        decision = "Reject the null hypothesis. There is a significant difference between the two samples."
    else:
        decision = "Fail to reject the null hypothesis. There is no significant difference between the two samples."

    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")
    print(decision)
    print("Assumptions:\n"
        "- The two samples are independent.\n"
        "- The two samples are normally distributed.\n"
        "- The two samples have equal variances.\n"
        "- The two samples have the same size.")
    
    return t_stat, p_value, decision

# # Example usage:
# sample1 = [25, 30, 35, 40, 45]
# sample2 = [35, 40, 45, 50, 55]
# t_stat, p_value, decision = two_sample_ttest(sample1, sample2)
# print(f"t-statistic: {t_stat}")
# print(f"p-value: {p_value}")
# print(decision)
# print("Assumptions:\n"
#       "- The two samples are independent.\n"
#       "- The two samples are normally distributed.\n"
#       "- The two samples have equal variances.\n"
#       "- The two samples have the same size.")

