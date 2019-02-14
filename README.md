This is a machine learning library developed by Alex Chen for CS5350/6350

Instructions for running decision tree:
    
    1. TO RUN:
        cd /path/to/DecisionTrees
        python3.7 Decision-Tree.py
        
    2. You can add methods into main(), but these first 3 are essential:
        
        Parameters:
        attributes, ordered_atts, numeric_atts, atts_with_unknown_val = read_txt_set_attr("DATA-DESC FILE CHANGE ME",
                                                                                      TREAT VALUE 'UNKNOWN' AS MISSING)
        numeric_atts_copy = numeric_atts.copy()
        training_data = read_csv("TRAIN CSV FILE", ordered_atts, numeric_atts, atts_with_unknown_val, TREAT VALUE 'UNKNOWN' AS MISSING)
        test_data = read_csv("TEST CSV FILE", ordered_atts, numeric_atts_copy, atts_with_unknown_val, TREAT VALUE 'UNKNOWN' AS MISSING)
    
        e.g.
        attributes, ordered_atts, numeric_atts, atts_with_unknown_val = read_txt_set_attr("bank/data-desc-readable.txt",
                                                                                      True)
        numeric_atts_copy = numeric_atts.copy()
        training_data = read_csv("bank/train.csv", ordered_atts, numeric_atts, atts_with_unknown_val, True)
        test_data = read_csv("bank/test.csv", ordered_atts, numeric_atts_copy, atts_with_unknown_val, True) 
    
    3. To create a tree
        Parameters:
            root = id3(training_set, attributes, depth, max_depth, gain_method)
        
        Max depth = 6 
        calculating ENTROPY
            root = id3(training_data, attributes, 0, 6, calc_entropy)
        ME
            root = id3(training_data, attributes, 0, 6, calc_majority_error)
        GI
            root = id3(training_data, attributes, 0, 6, calc_gini)
        
        You can print the tree with graphviz, too:
            draw_tree(root, "FILENAME")
        
    4.  Training-set accuracy vs Test-set accuracy
        for different gain methods and depths 
        (Prints to console)
        Parameters:
        find_average_accuracy_different_max_depths(training_data, test_data, attributes, num_trials, max_tree_depth)
        e.g.
        find_average_accuracy_different_max_depths(training_data, test_data, attributes, 10, 16)
    
            
        