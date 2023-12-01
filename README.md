## Getting Start
ALCS: A A clustering-based active learning method to query informative and representative samples

Here we prepared two versions of ALCS framework:
1. main_final_active2.py: implmentation of ALCS framework without the diversity exploration procedure.
2. main_final_active2.py: implementation of ALCS framework with the FPN-based diversity exploration procedure.

For both of them, you can change the label ratio by variable "label_ratiovalues". Also, you can substitute the fps_clustering function with any other clustering approach as long as it returns the following output variables:

	cluster centers
 	cluster labels
  	data samples
	class labels

## Example Usage:

To modify the input dataset, you can change the Input function in the code as follows:

	`sample = pd.read_csv('path/to/datasets)`

Also, an example jupternotebook file "example_alcs.ipynb" is added for users to directly use the code on google colab.

## Dependencies:

Please install the following package before running the python code.
* Numpy
* Pandas
* Scipy
* Scikit-learn

## Citation format:

For any use of this python code, please cite the following paper:
* Yan, Xuyang, Shabnam Nazmi, Biniam Gebru, Mohd Anwar, Abdollah Homaifar, Mrinmoy Sarkar, and Kishor Datta Gupta. "A clustering-based active learning method to query informative and representative samples." Applied Intelligence (2022): 1-18.
* Yan, X., Nazmi, S., Gebru, B., Anwar, M., Homaifar, A., Sarkar, M., & Gupta, K. D. (2022). Mitigating shortage of labeled data using clustering-based active learning with diversity exploration. arXiv preprint arXiv:2207.02964.
Note: the implementation of ALCS without diversity exploration will be updated soon to make the code easier to read and follow.

The bib style can be found below:
* `@article{yan2022clustering,
  title={A clustering-based active learning method to
  query informative and representative samples},
  author={Yan, Xuyang and Nazmi, Shabnam and Gebru,
  Biniam and Anwar, Mohd and Homaifar, Abdollah and Sarkar,
  Mrinmoy and Gupta, Kishor Datta},
  journal={Applied Intelligence},
  volume={52},
  number={11},
  pages={13250--13267},
  year={2022},
  publisher={Springer}
}`
* `@article{yan2022mitigating,
  title={Mitigating shortage of labeled data using
  clustering-based active learning with diversity exploration},
  author={Yan, Xuyang and Nazmi, Shabnam
  and Gebru, Biniam and Anwar, Mohd and Homaifar,
  Abdollah and Sarkar, Mrinmoy and Gupta, Kishor Datta},
  journal={arXiv preprint arXiv:2207.02964},
  year={2022}
}`
