This is a machine learning library developed by Alex Chen for CS5350/6350

Instructions for running decision tree:
    
    DT_1. TO RUN:
        cd /path/to/DecisionTrees
        python3.7 Decision-Tree.py
        
    DT_2. You can add methods into main:
	To read in data, call the following method
        
        attributes, training_data, test_data = get_atts_and_test_and_training_data_from_file(path/to/labels, path/to/train, path/to/test)
		optional parameter: 
			fill_unknown - fills "unknown" values with most common value of feature
				       default: False
	
	

        To create a tree
        root = make_d_tree(training_data, attributes)
		optinal parameters:
			gain - the information gain method. There are three options: calc_entropy, calc_gini, calc_majority_error
			       default: calc_entropy
			max_depth - sets maximum depth of tree
			       default: np.inf
			weights - array corresponding to each training sample's weight.  Used primarily for Ada boosting.
			       default: None 
			att_subset_size - int, randomly nominate n features to split on for each node. Used primarily for Random Forests 
			       default: -1
        
        To predict a test sample:
	get_predicted_label_from_tree(root, sample)

    DT_3.  Training-set accuracy vs Test-set accuracy for different gain methods and depths 
        find_average_accuracy_different_max_depths(training_data, test_data, attributes, num_trials, max_tree_depth)
        e.g.
        find_average_accuracy_different_max_depths(training_data, test_data, attributes, 10, 16)

Ensemble Learning:
    EL_1. TO RUN:
        cd /path/to/EnsembleLearning
        python3.7 AdaBoost.py

    EL_2. You can add methods into main:
	Call first method in DT_2 to import data
	
	train_err, test_err = error_vs_num_trees(ensemble_method - create_ada_boosted_stumps
					   			 - bagged_trees_or_rand_forest
			   				num_iter - number of trees
			   				attributes
			   				training_data
			   				test_data
			   
			   				optional parameter:
							att_subset_size - int, randomly nominates n features to split on for each node
									default: -1
						)

	Make figure from this data:
	
		x = [i + 1 for i in range(num_iter)]
	make_figure(x, train_err, test_err, graph_title, x_label, y_label, picture_name)


Linear Regression:
    LR_1. TO RUN:
	cd /path/to/LinearRegression
	python3.7 GradientDescent.py

    LR_2. You can add methods to main:
	train = read_csv(path/to/train)
	test = read_csv(path/to/test)
    	atts = read_txt(path/to/labels)
	
	To plot gradient descent's cost function with respect to the number of iterations:
	Prints resulting vector w cost function on test data too

	plot_descent(descent method - batch_gradient_descent OR random_stochastic_gradient_descent
				  r - the learning rate
			  threshold - For batch, when ||w_t+1 - w_t|| < threshold STOP
			   num_iter - terminates after n iterations
		      training_data
		          test_data
			       atts
			 plot_title
                            x_label
			    y_label
			   pic_name
		    )

Perceptron:
   P_1. TO RUN:
	cd /path/to/Perceptron
	python3.7 Perceptron.py

   P_2. You can add methods to main:
	train = read_csv(relative/path/to/train)
	test = read_csv(relative/path/to/test)
	
	To get weight vector or list of weight vectors, run perceptron algorithm:
	Standard:
	weight = perceptron(train_data, num_epochs, learn_rate, percep_type=0)
	Voted:
	weights_list = perceptron(train_data, num_epochs, learn_rate, percep_type=1)
	Average:
	weight, weights_list = perceptron(train_data, num_epochs, learn_rate, percep_type=2)

	To get error, of resulting weight vector or list:
	Standard and Average:
	find_error_on_test_data(weight, test_data)
	Voted:
	find_error_on_test_data(weights_list, test_data)
	
            
        
