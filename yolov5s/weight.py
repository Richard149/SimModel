import torch
from yolov5s import YOLOv5s

name_mapping = {
    # ===== model.0 → focus_0 =====
    'model.0.conv.conv.weight':                    'focus_0.conv.conv.weight',
    'model.0.conv.bn.weight':                      'focus_0.conv.bn.weight',
    'model.0.conv.bn.bias':                        'focus_0.conv.bn.bias',
    'model.0.conv.bn.running_mean':                'focus_0.conv.bn.running_mean',
    'model.0.conv.bn.running_var':                 'focus_0.conv.bn.running_var',
    'model.0.conv.bn.num_batches_tracked':         'focus_0.conv.bn.num_batches_tracked',

    # ===== model.1 → conv_1 =====
    'model.1.conv.weight':                          'conv_1.conv.weight',
    'model.1.bn.weight':                            'conv_1.bn.weight',
    'model.1.bn.bias':                              'conv_1.bn.bias',
    'model.1.bn.running_mean':                      'conv_1.bn.running_mean',
    'model.1.bn.running_var':                       'conv_1.bn.running_var',
    'model.1.bn.num_batches_tracked':               'conv_1.bn.num_batches_tracked',

    # ===== model.2 → c3_2 =====
    'model.2.cv1.conv.weight':                      'c3_2.cv1.conv.weight',
    'model.2.cv1.bn.weight':                        'c3_2.cv1.bn.weight',
    'model.2.cv1.bn.bias':                          'c3_2.cv1.bn.bias',
    'model.2.cv1.bn.running_mean':                  'c3_2.cv1.bn.running_mean',
    'model.2.cv1.bn.running_var':                   'c3_2.cv1.bn.running_var',
    'model.2.cv1.bn.num_batches_tracked':           'c3_2.cv1.bn.num_batches_tracked',
    'model.2.cv2.conv.weight':                      'c3_2.cv2.conv.weight',
    'model.2.cv2.bn.weight':                        'c3_2.cv2.bn.weight',
    'model.2.cv2.bn.bias':                          'c3_2.cv2.bn.bias',
    'model.2.cv2.bn.running_mean':                  'c3_2.cv2.bn.running_mean',
    'model.2.cv2.bn.running_var':                   'c3_2.cv2.bn.running_var',
    'model.2.cv2.bn.num_batches_tracked':           'c3_2.cv2.bn.num_batches_tracked',
    'model.2.cv3.conv.weight':                      'c3_2.cv3.conv.weight',
    'model.2.cv3.bn.weight':                        'c3_2.cv3.bn.weight',
    'model.2.cv3.bn.bias':                          'c3_2.cv3.bn.bias',
    'model.2.cv3.bn.running_mean':                  'c3_2.cv3.bn.running_mean',
    'model.2.cv3.bn.running_var':                   'c3_2.cv3.bn.running_var',
    'model.2.cv3.bn.num_batches_tracked':           'c3_2.cv3.bn.num_batches_tracked',
    'model.2.m.0.cv1.conv.weight':                  'c3_2.m.0.cv1.conv.weight',
    'model.2.m.0.cv1.bn.weight':                    'c3_2.m.0.cv1.bn.weight',
    'model.2.m.0.cv1.bn.bias':                      'c3_2.m.0.cv1.bn.bias',
    'model.2.m.0.cv1.bn.running_mean':              'c3_2.m.0.cv1.bn.running_mean',
    'model.2.m.0.cv1.bn.running_var':               'c3_2.m.0.cv1.bn.running_var',
    'model.2.m.0.cv1.bn.num_batches_tracked':       'c3_2.m.0.cv1.bn.num_batches_tracked',
    'model.2.m.0.cv2.conv.weight':                  'c3_2.m.0.cv2.conv.weight',
    'model.2.m.0.cv2.bn.weight':                    'c3_2.m.0.cv2.bn.weight',
    'model.2.m.0.cv2.bn.bias':                      'c3_2.m.0.cv2.bn.bias',
    'model.2.m.0.cv2.bn.running_mean':              'c3_2.m.0.cv2.bn.running_mean',
    'model.2.m.0.cv2.bn.running_var':               'c3_2.m.0.cv2.bn.running_var',
    'model.2.m.0.cv2.bn.num_batches_tracked':       'c3_2.m.0.cv2.bn.num_batches_tracked',

    # ===== model.3 → conv_3 =====
    'model.3.conv.weight':                          'conv_3.conv.weight',
    'model.3.bn.weight':                            'conv_3.bn.weight',
    'model.3.bn.bias':                              'conv_3.bn.bias',
    'model.3.bn.running_mean':                      'conv_3.bn.running_mean',
    'model.3.bn.running_var':                       'conv_3.bn.running_var',
    'model.3.bn.num_batches_tracked':               'conv_3.bn.num_batches_tracked',

    # ===== model.4 → c3_4 =====
    'model.4.cv1.conv.weight':                      'c3_4.cv1.conv.weight',
    'model.4.cv1.bn.weight':                        'c3_4.cv1.bn.weight',
    'model.4.cv1.bn.bias':                          'c3_4.cv1.bn.bias',
    'model.4.cv1.bn.running_mean':                  'c3_4.cv1.bn.running_mean',
    'model.4.cv1.bn.running_var':                   'c3_4.cv1.bn.running_var',
    'model.4.cv1.bn.num_batches_tracked':           'c3_4.cv1.bn.num_batches_tracked',
    'model.4.cv2.conv.weight':                      'c3_4.cv2.conv.weight',
    'model.4.cv2.bn.weight':                        'c3_4.cv2.bn.weight',
    'model.4.cv2.bn.bias':                          'c3_4.cv2.bn.bias',
    'model.4.cv2.bn.running_mean':                  'c3_4.cv2.bn.running_mean',
    'model.4.cv2.bn.running_var':                   'c3_4.cv2.bn.running_var',
    'model.4.cv2.bn.num_batches_tracked':           'c3_4.cv2.bn.num_batches_tracked',
    'model.4.cv3.conv.weight':                      'c3_4.cv3.conv.weight',
    'model.4.cv3.bn.weight':                        'c3_4.cv3.bn.weight',
    'model.4.cv3.bn.bias':                          'c3_4.cv3.bn.bias',
    'model.4.cv3.bn.running_mean':                  'c3_4.cv3.bn.running_mean',
    'model.4.cv3.bn.running_var':                   'c3_4.cv3.bn.running_var',
    'model.4.cv3.bn.num_batches_tracked':           'c3_4.cv3.bn.num_batches_tracked',
    'model.4.m.0.cv1.conv.weight':                  'c3_4.m.0.cv1.conv.weight',
    'model.4.m.0.cv1.bn.weight':                    'c3_4.m.0.cv1.bn.weight',
    'model.4.m.0.cv1.bn.bias':                      'c3_4.m.0.cv1.bn.bias',
    'model.4.m.0.cv1.bn.running_mean':              'c3_4.m.0.cv1.bn.running_mean',
    'model.4.m.0.cv1.bn.running_var':               'c3_4.m.0.cv1.bn.running_var',
    'model.4.m.0.cv1.bn.num_batches_tracked':       'c3_4.m.0.cv1.bn.num_batches_tracked',
    'model.4.m.0.cv2.conv.weight':                  'c3_4.m.0.cv2.conv.weight',
    'model.4.m.0.cv2.bn.weight':                    'c3_4.m.0.cv2.bn.weight',
    'model.4.m.0.cv2.bn.bias':                      'c3_4.m.0.cv2.bn.bias',
    'model.4.m.0.cv2.bn.running_mean':              'c3_4.m.0.cv2.bn.running_mean',
    'model.4.m.0.cv2.bn.running_var':               'c3_4.m.0.cv2.bn.running_var',
    'model.4.m.0.cv2.bn.num_batches_tracked':       'c3_4.m.0.cv2.bn.num_batches_tracked',
    'model.4.m.1.cv1.conv.weight':                  'c3_4.m.1.cv1.conv.weight',
    'model.4.m.1.cv1.bn.weight':                    'c3_4.m.1.cv1.bn.weight',
    'model.4.m.1.cv1.bn.bias':                      'c3_4.m.1.cv1.bn.bias',
    'model.4.m.1.cv1.bn.running_mean':              'c3_4.m.1.cv1.bn.running_mean',
    'model.4.m.1.cv1.bn.running_var':               'c3_4.m.1.cv1.bn.running_var',
    'model.4.m.1.cv1.bn.num_batches_tracked':       'c3_4.m.1.cv1.bn.num_batches_tracked',
    'model.4.m.1.cv2.conv.weight':                  'c3_4.m.1.cv2.conv.weight',
    'model.4.m.1.cv2.bn.weight':                    'c3_4.m.1.cv2.bn.weight',
    'model.4.m.1.cv2.bn.bias':                      'c3_4.m.1.cv2.bn.bias',
    'model.4.m.1.cv2.bn.running_mean':              'c3_4.m.1.cv2.bn.running_mean',
    'model.4.m.1.cv2.bn.running_var':               'c3_4.m.1.cv2.bn.running_var',
    'model.4.m.1.cv2.bn.num_batches_tracked':       'c3_4.m.1.cv2.bn.num_batches_tracked',
    'model.4.m.2.cv1.conv.weight':                  'c3_4.m.2.cv1.conv.weight',
    'model.4.m.2.cv1.bn.weight':                    'c3_4.m.2.cv1.bn.weight',
    'model.4.m.2.cv1.bn.bias':                      'c3_4.m.2.cv1.bn.bias',
    'model.4.m.2.cv1.bn.running_mean':              'c3_4.m.2.cv1.bn.running_mean',
    'model.4.m.2.cv1.bn.running_var':               'c3_4.m.2.cv1.bn.running_var',
    'model.4.m.2.cv1.bn.num_batches_tracked':       'c3_4.m.2.cv1.bn.num_batches_tracked',
    'model.4.m.2.cv2.conv.weight':                  'c3_4.m.2.cv2.conv.weight',
    'model.4.m.2.cv2.bn.weight':                    'c3_4.m.2.cv2.bn.weight',
    'model.4.m.2.cv2.bn.bias':                      'c3_4.m.2.cv2.bn.bias',
    'model.4.m.2.cv2.bn.running_mean':              'c3_4.m.2.cv2.bn.running_mean',
    'model.4.m.2.cv2.bn.running_var':               'c3_4.m.2.cv2.bn.running_var',
    'model.4.m.2.cv2.bn.num_batches_tracked':       'c3_4.m.2.cv2.bn.num_batches_tracked',

    # ===== model.5 → conv_5 =====
    'model.5.conv.weight':                          'conv_5.conv.weight',
    'model.5.bn.weight':                            'conv_5.bn.weight',
    'model.5.bn.bias':                              'conv_5.bn.bias',
    'model.5.bn.running_mean':                      'conv_5.bn.running_mean',
    'model.5.bn.running_var':                       'conv_5.bn.running_var',
    'model.5.bn.num_batches_tracked':               'conv_5.bn.num_batches_tracked',

    # ===== model.6 → c3_6 =====
    'model.6.cv1.conv.weight':                      'c3_6.cv1.conv.weight',
    'model.6.cv1.bn.weight':                        'c3_6.cv1.bn.weight',
    'model.6.cv1.bn.bias':                          'c3_6.cv1.bn.bias',
    'model.6.cv1.bn.running_mean':                  'c3_6.cv1.bn.running_mean',
    'model.6.cv1.bn.running_var':                   'c3_6.cv1.bn.running_var',
    'model.6.cv1.bn.num_batches_tracked':           'c3_6.cv1.bn.num_batches_tracked',
    'model.6.cv2.conv.weight':                      'c3_6.cv2.conv.weight',
    'model.6.cv2.bn.weight':                        'c3_6.cv2.bn.weight',
    'model.6.cv2.bn.bias':                          'c3_6.cv2.bn.bias',
    'model.6.cv2.bn.running_mean':                  'c3_6.cv2.bn.running_mean',
    'model.6.cv2.bn.running_var':                   'c3_6.cv2.bn.running_var',
    'model.6.cv2.bn.num_batches_tracked':           'c3_6.cv2.bn.num_batches_tracked',
    'model.6.cv3.conv.weight':                      'c3_6.cv3.conv.weight',
    'model.6.cv3.bn.weight':                        'c3_6.cv3.bn.weight',
    'model.6.cv3.bn.bias':                          'c3_6.cv3.bn.bias',
    'model.6.cv3.bn.running_mean':                  'c3_6.cv3.bn.running_mean',
    'model.6.cv3.bn.running_var':                   'c3_6.cv3.bn.running_var',
    'model.6.cv3.bn.num_batches_tracked':           'c3_6.cv3.bn.num_batches_tracked',
    'model.6.m.0.cv1.conv.weight':                  'c3_6.m.0.cv1.conv.weight',
    'model.6.m.0.cv1.bn.weight':                    'c3_6.m.0.cv1.bn.weight',
    'model.6.m.0.cv1.bn.bias':                      'c3_6.m.0.cv1.bn.bias',
    'model.6.m.0.cv1.bn.running_mean':              'c3_6.m.0.cv1.bn.running_mean',
    'model.6.m.0.cv1.bn.running_var':               'c3_6.m.0.cv1.bn.running_var',
    'model.6.m.0.cv1.bn.num_batches_tracked':       'c3_6.m.0.cv1.bn.num_batches_tracked',
    'model.6.m.0.cv2.conv.weight':                  'c3_6.m.0.cv2.conv.weight',
    'model.6.m.0.cv2.bn.weight':                    'c3_6.m.0.cv2.bn.weight',
    'model.6.m.0.cv2.bn.bias':                      'c3_6.m.0.cv2.bn.bias',
    'model.6.m.0.cv2.bn.running_mean':              'c3_6.m.0.cv2.bn.running_mean',
    'model.6.m.0.cv2.bn.running_var':               'c3_6.m.0.cv2.bn.running_var',
    'model.6.m.0.cv2.bn.num_batches_tracked':       'c3_6.m.0.cv2.bn.num_batches_tracked',
    'model.6.m.1.cv1.conv.weight':                  'c3_6.m.1.cv1.conv.weight',
    'model.6.m.1.cv1.bn.weight':                    'c3_6.m.1.cv1.bn.weight',
    'model.6.m.1.cv1.bn.bias':                      'c3_6.m.1.cv1.bn.bias',
    'model.6.m.1.cv1.bn.running_mean':              'c3_6.m.1.cv1.bn.running_mean',
    'model.6.m.1.cv1.bn.running_var':               'c3_6.m.1.cv1.bn.running_var',
    'model.6.m.1.cv1.bn.num_batches_tracked':       'c3_6.m.1.cv1.bn.num_batches_tracked',
    'model.6.m.1.cv2.conv.weight':                  'c3_6.m.1.cv2.conv.weight',
    'model.6.m.1.cv2.bn.weight':                    'c3_6.m.1.cv2.bn.weight',
    'model.6.m.1.cv2.bn.bias':                      'c3_6.m.1.cv2.bn.bias',
    'model.6.m.1.cv2.bn.running_mean':              'c3_6.m.1.cv2.bn.running_mean',
    'model.6.m.1.cv2.bn.running_var':               'c3_6.m.1.cv2.bn.running_var',
    'model.6.m.1.cv2.bn.num_batches_tracked':       'c3_6.m.1.cv2.bn.num_batches_tracked',
    'model.6.m.2.cv1.conv.weight':                  'c3_6.m.2.cv1.conv.weight',
    'model.6.m.2.cv1.bn.weight':                    'c3_6.m.2.cv1.bn.weight',
    'model.6.m.2.cv1.bn.bias':                      'c3_6.m.2.cv1.bn.bias',
    'model.6.m.2.cv1.bn.running_mean':              'c3_6.m.2.cv1.bn.running_mean',
    'model.6.m.2.cv1.bn.running_var':               'c3_6.m.2.cv1.bn.running_var',
    'model.6.m.2.cv1.bn.num_batches_tracked':       'c3_6.m.2.cv1.bn.num_batches_tracked',
    'model.6.m.2.cv2.conv.weight':                  'c3_6.m.2.cv2.conv.weight',
    'model.6.m.2.cv2.bn.weight':                    'c3_6.m.2.cv2.bn.weight',
    'model.6.m.2.cv2.bn.bias':                      'c3_6.m.2.cv2.bn.bias',
    'model.6.m.2.cv2.bn.running_mean':              'c3_6.m.2.cv2.bn.running_mean',
    'model.6.m.2.cv2.bn.running_var':               'c3_6.m.2.cv2.bn.running_var',
    'model.6.m.2.cv2.bn.num_batches_tracked':       'c3_6.m.2.cv2.bn.num_batches_tracked',

    # ===== model.7 → conv_7 =====
    'model.7.conv.weight':                          'conv_7.conv.weight',
    'model.7.bn.weight':                            'conv_7.bn.weight',
    'model.7.bn.bias':                              'conv_7.bn.bias',
    'model.7.bn.running_mean':                      'conv_7.bn.running_mean',
    'model.7.bn.running_var':                       'conv_7.bn.running_var',
    'model.7.bn.num_batches_tracked':               'conv_7.bn.num_batches_tracked',

    # ===== model.8 → spp_8 =====
    'model.8.cv1.conv.weight':                      'spp_8.cv1.conv.weight',
    'model.8.cv1.bn.weight':                        'spp_8.cv1.bn.weight',
    'model.8.cv1.bn.bias':                          'spp_8.cv1.bn.bias',
    'model.8.cv1.bn.running_mean':                  'spp_8.cv1.bn.running_mean',
    'model.8.cv1.bn.running_var':                   'spp_8.cv1.bn.running_var',
    'model.8.cv1.bn.num_batches_tracked':           'spp_8.cv1.bn.num_batches_tracked',
    'model.8.cv2.conv.weight':                      'spp_8.cv2.conv.weight',
    'model.8.cv2.bn.weight':                        'spp_8.cv2.bn.weight',
    'model.8.cv2.bn.bias':                          'spp_8.cv2.bn.bias',
    'model.8.cv2.bn.running_mean':                  'spp_8.cv2.bn.running_mean',
    'model.8.cv2.bn.running_var':                   'spp_8.cv2.bn.running_var',
    'model.8.cv2.bn.num_batches_tracked':           'spp_8.cv2.bn.num_batches_tracked',

    # ===== model.9 → c3_9 =====
    'model.9.cv1.conv.weight':                      'c3_9.cv1.conv.weight',
    'model.9.cv1.bn.weight':                        'c3_9.cv1.bn.weight',
    'model.9.cv1.bn.bias':                          'c3_9.cv1.bn.bias',
    'model.9.cv1.bn.running_mean':                  'c3_9.cv1.bn.running_mean',
    'model.9.cv1.bn.running_var':                   'c3_9.cv1.bn.running_var',
    'model.9.cv1.bn.num_batches_tracked':           'c3_9.cv1.bn.num_batches_tracked',
    'model.9.cv2.conv.weight':                      'c3_9.cv2.conv.weight',
    'model.9.cv2.bn.weight':                        'c3_9.cv2.bn.weight',
    'model.9.cv2.bn.bias':                          'c3_9.cv2.bn.bias',
    'model.9.cv2.bn.running_mean':                  'c3_9.cv2.bn.running_mean',
    'model.9.cv2.bn.running_var':                   'c3_9.cv2.bn.running_var',
    'model.9.cv2.bn.num_batches_tracked':           'c3_9.cv2.bn.num_batches_tracked',
    'model.9.cv3.conv.weight':                      'c3_9.cv3.conv.weight',
    'model.9.cv3.bn.weight':                        'c3_9.cv3.bn.weight',
    'model.9.cv3.bn.bias':                          'c3_9.cv3.bn.bias',
    'model.9.cv3.bn.running_mean':                  'c3_9.cv3.bn.running_mean',
    'model.9.cv3.bn.running_var':                   'c3_9.cv3.bn.running_var',
    'model.9.cv3.bn.num_batches_tracked':           'c3_9.cv3.bn.num_batches_tracked',
    'model.9.m.0.cv1.conv.weight':                  'c3_9.m.0.cv1.conv.weight',
    'model.9.m.0.cv1.bn.weight':                    'c3_9.m.0.cv1.bn.weight',
    'model.9.m.0.cv1.bn.bias':                      'c3_9.m.0.cv1.bn.bias',
    'model.9.m.0.cv1.bn.running_mean':              'c3_9.m.0.cv1.bn.running_mean',
    'model.9.m.0.cv1.bn.running_var':               'c3_9.m.0.cv1.bn.running_var',
    'model.9.m.0.cv1.bn.num_batches_tracked':       'c3_9.m.0.cv1.bn.num_batches_tracked',
    'model.9.m.0.cv2.conv.weight':                  'c3_9.m.0.cv2.conv.weight',
    'model.9.m.0.cv2.bn.weight':                    'c3_9.m.0.cv2.bn.weight',
    'model.9.m.0.cv2.bn.bias':                      'c3_9.m.0.cv2.bn.bias',
    'model.9.m.0.cv2.bn.running_mean':              'c3_9.m.0.cv2.bn.running_mean',
    'model.9.m.0.cv2.bn.running_var':               'c3_9.m.0.cv2.bn.running_var',
    'model.9.m.0.cv2.bn.num_batches_tracked':       'c3_9.m.0.cv2.bn.num_batches_tracked',

    # ===== model.10 → conv_10 =====
    'model.10.conv.weight':                          'conv_10.conv.weight',
    'model.10.bn.weight':                            'conv_10.bn.weight',
    'model.10.bn.bias':                              'conv_10.bn.bias',
    'model.10.bn.running_mean':                      'conv_10.bn.running_mean',
    'model.10.bn.running_var':                       'conv_10.bn.running_var',
    'model.10.bn.num_batches_tracked':               'conv_10.bn.num_batches_tracked',

    # ===== model.13 → c3_13 =====
    'model.13.cv1.conv.weight':                      'c3_13.cv1.conv.weight',
    'model.13.cv1.bn.weight':                        'c3_13.cv1.bn.weight',
    'model.13.cv1.bn.bias':                          'c3_13.cv1.bn.bias',
    'model.13.cv1.bn.running_mean':                  'c3_13.cv1.bn.running_mean',
    'model.13.cv1.bn.running_var':                   'c3_13.cv1.bn.running_var',
    'model.13.cv1.bn.num_batches_tracked':           'c3_13.cv1.bn.num_batches_tracked',
    'model.13.cv2.conv.weight':                      'c3_13.cv2.conv.weight',
    'model.13.cv2.bn.weight':                        'c3_13.cv2.bn.weight',
    'model.13.cv2.bn.bias':                          'c3_13.cv2.bn.bias',
    'model.13.cv2.bn.running_mean':                  'c3_13.cv2.bn.running_mean',
    'model.13.cv2.bn.running_var':                   'c3_13.cv2.bn.running_var',
    'model.13.cv2.bn.num_batches_tracked':           'c3_13.cv2.bn.num_batches_tracked',
    'model.13.cv3.conv.weight':                      'c3_13.cv3.conv.weight',
    'model.13.cv3.bn.weight':                        'c3_13.cv3.bn.weight',
    'model.13.cv3.bn.bias':                          'c3_13.cv3.bn.bias',
    'model.13.cv3.bn.running_mean':                  'c3_13.cv3.bn.running_mean',
    'model.13.cv3.bn.running_var':                   'c3_13.cv3.bn.running_var',
    'model.13.cv3.bn.num_batches_tracked':           'c3_13.cv3.bn.num_batches_tracked',
    'model.13.m.0.cv1.conv.weight':                  'c3_13.m.0.cv1.conv.weight',
    'model.13.m.0.cv1.bn.weight':                    'c3_13.m.0.cv1.bn.weight',
    'model.13.m.0.cv1.bn.bias':                      'c3_13.m.0.cv1.bn.bias',
    'model.13.m.0.cv1.bn.running_mean':              'c3_13.m.0.cv1.bn.running_mean',
    'model.13.m.0.cv1.bn.running_var':               'c3_13.m.0.cv1.bn.running_var',
    'model.13.m.0.cv1.bn.num_batches_tracked':       'c3_13.m.0.cv1.bn.num_batches_tracked',
    'model.13.m.0.cv2.conv.weight':                  'c3_13.m.0.cv2.conv.weight',
    'model.13.m.0.cv2.bn.weight':                    'c3_13.m.0.cv2.bn.weight',
    'model.13.m.0.cv2.bn.bias':                      'c3_13.m.0.cv2.bn.bias',
    'model.13.m.0.cv2.bn.running_mean':              'c3_13.m.0.cv2.bn.running_mean',
    'model.13.m.0.cv2.bn.running_var':               'c3_13.m.0.cv2.bn.running_var',
    'model.13.m.0.cv2.bn.num_batches_tracked':       'c3_13.m.0.cv2.bn.num_batches_tracked',

    # ===== model.14 → conv_14 =====
    'model.14.conv.weight':                           'conv_14.conv.weight',
    'model.14.bn.weight':                             'conv_14.bn.weight',
    'model.14.bn.bias':                               'conv_14.bn.bias',
    'model.14.bn.running_mean':                       'conv_14.bn.running_mean',
    'model.14.bn.running_var':                        'conv_14.bn.running_var',
    'model.14.bn.num_batches_tracked':                'conv_14.bn.num_batches_tracked',

    # ===== model.17 → c3_17 =====
    'model.17.cv1.conv.weight':                       'c3_17.cv1.conv.weight',
    'model.17.cv1.bn.weight':                         'c3_17.cv1.bn.weight',
    'model.17.cv1.bn.bias':                           'c3_17.cv1.bn.bias',
    'model.17.cv1.bn.running_mean':                   'c3_17.cv1.bn.running_mean',
    'model.17.cv1.bn.running_var':                    'c3_17.cv1.bn.running_var',
    'model.17.cv1.bn.num_batches_tracked':            'c3_17.cv1.bn.num_batches_tracked',
    'model.17.cv2.conv.weight':                       'c3_17.cv2.conv.weight',
    'model.17.cv2.bn.weight':                         'c3_17.cv2.bn.weight',
    'model.17.cv2.bn.bias':                           'c3_17.cv2.bn.bias',
    'model.17.cv2.bn.running_mean':                   'c3_17.cv2.bn.running_mean',
    'model.17.cv2.bn.running_var':                    'c3_17.cv2.bn.running_var',
    'model.17.cv2.bn.num_batches_tracked':            'c3_17.cv2.bn.num_batches_tracked',
    'model.17.cv3.conv.weight':                       'c3_17.cv3.conv.weight',
    'model.17.cv3.bn.weight':                         'c3_17.cv3.bn.weight',
    'model.17.cv3.bn.bias':                           'c3_17.cv3.bn.bias',
    'model.17.cv3.bn.running_mean':                   'c3_17.cv3.bn.running_mean',
    'model.17.cv3.bn.running_var':                    'c3_17.cv3.bn.running_var',
    'model.17.cv3.bn.num_batches_tracked':            'c3_17.cv3.bn.num_batches_tracked',
    'model.17.m.0.cv1.conv.weight':                   'c3_17.m.0.cv1.conv.weight',
    'model.17.m.0.cv1.bn.weight':                     'c3_17.m.0.cv1.bn.weight',
    'model.17.m.0.cv1.bn.bias':                       'c3_17.m.0.cv1.bn.bias',
    'model.17.m.0.cv1.bn.running_mean':               'c3_17.m.0.cv1.bn.running_mean',
    'model.17.m.0.cv1.bn.running_var':                'c3_17.m.0.cv1.bn.running_var',
    'model.17.m.0.cv1.bn.num_batches_tracked':        'c3_17.m.0.cv1.bn.num_batches_tracked',
    'model.17.m.0.cv2.conv.weight':                   'c3_17.m.0.cv2.conv.weight',
    'model.17.m.0.cv2.bn.weight':                     'c3_17.m.0.cv2.bn.weight',
    'model.17.m.0.cv2.bn.bias':                       'c3_17.m.0.cv2.bn.bias',
    'model.17.m.0.cv2.bn.running_mean':               'c3_17.m.0.cv2.bn.running_mean',
    'model.17.m.0.cv2.bn.running_var':                'c3_17.m.0.cv2.bn.running_var',
    'model.17.m.0.cv2.bn.num_batches_tracked':        'c3_17.m.0.cv2.bn.num_batches_tracked',

    # ===== model.18 → conv_18 =====
    'model.18.conv.weight':                           'conv_18.conv.weight',
    'model.18.bn.weight':                             'conv_18.bn.weight',
    'model.18.bn.bias':                               'conv_18.bn.bias',
    'model.18.bn.running_mean':                       'conv_18.bn.running_mean',
    'model.18.bn.running_var':                        'conv_18.bn.running_var',
    'model.18.bn.num_batches_tracked':                'conv_18.bn.num_batches_tracked',

    # ===== model.20 → c3_20 =====
    'model.20.cv1.conv.weight':                       'c3_20.cv1.conv.weight',
    'model.20.cv1.bn.weight':                         'c3_20.cv1.bn.weight',
    'model.20.cv1.bn.bias':                           'c3_20.cv1.bn.bias',
    'model.20.cv1.bn.running_mean':                   'c3_20.cv1.bn.running_mean',
    'model.20.cv1.bn.running_var':                    'c3_20.cv1.bn.running_var',
    'model.20.cv1.bn.num_batches_tracked':            'c3_20.cv1.bn.num_batches_tracked',
    'model.20.cv2.conv.weight':                       'c3_20.cv2.conv.weight',
    'model.20.cv2.bn.weight':                         'c3_20.cv2.bn.weight',
    'model.20.cv2.bn.bias':                           'c3_20.cv2.bn.bias',
    'model.20.cv2.bn.running_mean':                   'c3_20.cv2.bn.running_mean',
    'model.20.cv2.bn.running_var':                    'c3_20.cv2.bn.running_var',
    'model.20.cv2.bn.num_batches_tracked':            'c3_20.cv2.bn.num_batches_tracked',
    'model.20.cv3.conv.weight':                       'c3_20.cv3.conv.weight',
    'model.20.cv3.bn.weight':                         'c3_20.cv3.bn.weight',
    'model.20.cv3.bn.bias':                           'c3_20.cv3.bn.bias',
    'model.20.cv3.bn.running_mean':                   'c3_20.cv3.bn.running_mean',
    'model.20.cv3.bn.running_var':                    'c3_20.cv3.bn.running_var',
    'model.20.cv3.bn.num_batches_tracked':            'c3_20.cv3.bn.num_batches_tracked',
    'model.20.m.0.cv1.conv.weight':                   'c3_20.m.0.cv1.conv.weight',
    'model.20.m.0.cv1.bn.weight':                     'c3_20.m.0.cv1.bn.weight',
    'model.20.m.0.cv1.bn.bias':                       'c3_20.m.0.cv1.bn.bias',
    'model.20.m.0.cv1.bn.running_mean':               'c3_20.m.0.cv1.bn.running_mean',
    'model.20.m.0.cv1.bn.running_var':                'c3_20.m.0.cv1.bn.running_var',
    'model.20.m.0.cv1.bn.num_batches_tracked':        'c3_20.m.0.cv1.bn.num_batches_tracked',
    'model.20.m.0.cv2.conv.weight':                   'c3_20.m.0.cv2.conv.weight',
    'model.20.m.0.cv2.bn.weight':                     'c3_20.m.0.cv2.bn.weight',
    'model.20.m.0.cv2.bn.bias':                       'c3_20.m.0.cv2.bn.bias',
    'model.20.m.0.cv2.bn.running_mean':               'c3_20.m.0.cv2.bn.running_mean',
    'model.20.m.0.cv2.bn.running_var':                'c3_20.m.0.cv2.bn.running_var',
    'model.20.m.0.cv2.bn.num_batches_tracked':        'c3_20.m.0.cv2.bn.num_batches_tracked',

    # ===== model.21 → conv_21 =====
    'model.21.conv.weight':                           'conv_21.conv.weight',
    'model.21.bn.weight':                             'conv_21.bn.weight',
    'model.21.bn.bias':                               'conv_21.bn.bias',
    'model.21.bn.running_mean':                       'conv_21.bn.running_mean',
    'model.21.bn.running_var':                        'conv_21.bn.running_var',
    'model.21.bn.num_batches_tracked':                'conv_21.bn.num_batches_tracked',

    # ===== model.23 → c3_23 =====
    'model.23.cv1.conv.weight':                       'c3_23.cv1.conv.weight',
    'model.23.cv1.bn.weight':                         'c3_23.cv1.bn.weight',
    'model.23.cv1.bn.bias':                           'c3_23.cv1.bn.bias',
    'model.23.cv1.bn.running_mean':                   'c3_23.cv1.bn.running_mean',
    'model.23.cv1.bn.running_var':                    'c3_23.cv1.bn.running_var',
    'model.23.cv1.bn.num_batches_tracked':            'c3_23.cv1.bn.num_batches_tracked',
    'model.23.cv2.conv.weight':                       'c3_23.cv2.conv.weight',
    'model.23.cv2.bn.weight':                         'c3_23.cv2.bn.weight',
    'model.23.cv2.bn.bias':                           'c3_23.cv2.bn.bias',
    'model.23.cv2.bn.running_mean':                   'c3_23.cv2.bn.running_mean',
    'model.23.cv2.bn.running_var':                    'c3_23.cv2.bn.running_var',
    'model.23.cv2.bn.num_batches_tracked':            'c3_23.cv2.bn.num_batches_tracked',
    'model.23.cv3.conv.weight':                       'c3_23.cv3.conv.weight',
    'model.23.cv3.bn.weight':                         'c3_23.cv3.bn.weight',
    'model.23.cv3.bn.bias':                           'c3_23.cv3.bn.bias',
    'model.23.cv3.bn.running_mean':                   'c3_23.cv3.bn.running_mean',
    'model.23.cv3.bn.running_var':                    'c3_23.cv3.bn.running_var',
    'model.23.cv3.bn.num_batches_tracked':            'c3_23.cv3.bn.num_batches_tracked',
    'model.23.m.0.cv1.conv.weight':                   'c3_23.m.0.cv1.conv.weight',
    'model.23.m.0.cv1.bn.weight':                     'c3_23.m.0.cv1.bn.weight',
    'model.23.m.0.cv1.bn.bias':                       'c3_23.m.0.cv1.bn.bias',
    'model.23.m.0.cv1.bn.running_mean':               'c3_23.m.0.cv1.bn.running_mean',
    'model.23.m.0.cv1.bn.running_var':                'c3_23.m.0.cv1.bn.running_var',
    'model.23.m.0.cv1.bn.num_batches_tracked':        'c3_23.m.0.cv1.bn.num_batches_tracked',
    'model.23.m.0.cv2.conv.weight':                   'c3_23.m.0.cv2.conv.weight',
    'model.23.m.0.cv2.bn.weight':                     'c3_23.m.0.cv2.bn.weight',
    'model.23.m.0.cv2.bn.bias':                       'c3_23.m.0.cv2.bn.bias',
    'model.23.m.0.cv2.bn.running_mean':               'c3_23.m.0.cv2.bn.running_mean',
    'model.23.m.0.cv2.bn.running_var':                'c3_23.m.0.cv2.bn.running_var',
    'model.23.m.0.cv2.bn.num_batches_tracked':        'c3_23.m.0.cv2.bn.num_batches_tracked',

    # ===== model.24 detect head → detect_layer_X_conv =====
    'model.24.m.0.weight':                             'detect_layer_0_conv.weight',
    'model.24.m.0.bias':                               'detect_layer_0_conv.bias',
    'model.24.m.1.weight':                             'detect_layer_1_conv.weight',
    'model.24.m.1.bias':                               'detect_layer_1_conv.bias',
    'model.24.m.2.weight':                             'detect_layer_2_conv.weight',
    'model.24.m.2.bias':                               'detect_layer_2_conv.bias',
}

def mapped_load(my_model, pure_dict):
    # 执行映射加载
    my_dict = my_model.state_dict()
    loaded = {}
    skipped = []
    mismatched = []
    for official_key, my_key in name_mapping.items():
        if official_key not in pure_dict:
            skipped.append(f"官方缺少: {official_key}")
            continue
        if my_key not in my_dict:
            skipped.append(f"模型缺少: {my_key}")
            continue
        if my_dict[my_key].shape != pure_dict[official_key].shape:
            mismatched.append(f"{my_key}: 模型{my_dict[my_key].shape} vs 官方{pure_dict[official_key].shape}")
            continue
        loaded[my_key] = pure_dict[official_key]

    my_model.load_state_dict(loaded, strict=False)
    print(f"成功加载: {len(loaded)}/{len(my_dict)} 个参数")
    if skipped:
        print(f"跳过 (key不存在): {len(skipped)}")
        for s in skipped[:10]:
            print(f"  {s}")
    if mismatched:
        print(f"形状不匹配: {len(mismatched)}")
        for m in mismatched[:10]:
            print(f"  {m}")
    
    unmatched = set(my_dict.keys()) - set(loaded.keys())
    print(f"未匹配: {len(unmatched)} 个")
    for k in sorted(unmatched)[:]:
        print(f"  {k}")

if __name__ == "__main__":
    model = YOLOv5s()
    pure_dict = torch.load('/home/lijiaxin/yolov5/yolov5s_state_dict.pt', 
                        map_location='cpu', weights_only=True)

    # 打印一个加载前模型的参数
    print("加载前模型参数:")
    print(model.state_dict()['c3_20.m.0.cv1.conv.weight'][0][0:2])

    mapped_load(model, pure_dict)

    print("加载后模型参数:")
    print(model.state_dict()['c3_20.m.0.cv1.conv.weight'][0][0:2])


