import numpy as np
import torch
semantic_kitti_class_frequencies = np.array([1.57835390e07,
                                                        1.25136000e05,
                                                        1.18809000e05,
                                                        6.46799000e05,
                                                        8.21951000e05,
                                                        2.62978000e05,
                                                        2.83696000e05,
                                                        2.04750000e05,
                                                        6.16887030e07,
                                                        4.50296100e06,
                                                        4.48836500e07,
                                                        2.26992300e06,
                                                        5.68402180e07,
                                                        1.57196520e07,
                                                        1.58442623e08,
                                                        2.06162300e06,
                                                        3.69705220e07,
                                                        1.15198800e06,
                                                        3.34146000e05,
                                                        5.41773033e09,  # empty
                                                        ])
class_weights = torch.from_numpy(1/np.log(semantic_kitti_class_frequencies + 0.001))
print(class_weights)
