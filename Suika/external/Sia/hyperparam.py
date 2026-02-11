from Suika.cluster.resource import default_acc_type, default_n_acc_per_node, default_n_node

# hyperparameters
k_acc_type = default_acc_type
k_n_acc_per_node = default_n_acc_per_node
k_max_n_acc = k_n_acc_per_node * default_n_node

k_p_fairness = -0.5  # Default value for p_fairness in Sia
k_lambda_a = 0.01  # Default value for lambda_a in Sia # doubt if value this small could make any difference; anyway, respect the artifact.
k_lambda_n = 1.1  # Default value for lambda_n in Sia

INF = 1000000

DETAILED_LOGGING = False