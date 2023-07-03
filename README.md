# ALCS: A A clustering-based active learning method to query informative and representative samples

Here we prepared two versions of ALCS framework:
1. main_final_active2.py: implmentation of ALCS framework without the diversity exploration procedure.
2. main_final_active2.py: implementation of ALCS framework with the FPN-based diversity exploration procedure.

For both of them, you can change the label ratio by variable "label_ratiovalues". Also, you can substitute the fps_clustering function with any other clustering approach as long as it returns the following output variables:

	cluster centers
 	cluster labels
  	data samples
	class labels

To modify the input dataset, you can change the Input function in the code as follows:

	sample = pd.read_csv('path/to/datasets)

For any use of this python code, please cite the following paper:
	
1.  Yan, Xuyang, Shabnam Nazmi, Biniam Gebru, Mohd Anwar, Abdollah Homaifar, Mrinmoy Sarkar, and Kishor Datta Gupta. "A clustering-based active learning method to query informative and representative samples." Applied Intelligence (2022): 1-18.

Note: the implementation of ALCS without diversity exploration will be updated soon to make the code easier to read and follow.
