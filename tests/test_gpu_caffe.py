import tempfile


def test_caffe_gpu():
    # adapted from caffe's test suite
    # https://github.com/BVLC/caffe/blob/master/python/caffe/test/test_solver.py
    import os

    import numpy as np

    import caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()

    def simple_net_file(num_output):
        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write("""name: 'testnet' force_backward: true
        layer { type: 'DummyData' name: 'data' top: 'data' top: 'label'
        dummy_data_param { num: 5 channels: 2 height: 3 width: 4
            num: 5 channels: 1 height: 1 width: 1
            data_filler { type: 'gaussian' std: 1 }
            data_filler { type: 'constant' } } }
        layer { type: 'Convolution' name: 'conv' bottom: 'data' top: 'conv'
        convolution_param { num_output: 11 kernel_size: 2 pad: 3
            weight_filler { type: 'gaussian' std: 1 }
            bias_filler { type: 'constant' value: 2 } }
            param { decay_mult: 1 } param { decay_mult: 0 }
            }
        layer { type: 'InnerProduct' name: 'ip' bottom: 'conv' top: 'ip_blob'
        inner_product_param { num_output: """ + str(num_output) + """
            weight_filler { type: 'gaussian' std: 2.5 }
            bias_filler { type: 'constant' value: -3 } } }
        layer { type: 'SoftmaxWithLoss' name: 'loss' bottom: 'ip_blob'
                bottom: 'label'
        top: 'loss' }""")
        f.close()
        return f.name

    num_output = 13
    net_f = simple_net_file(num_output)
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""net: '""" + net_f + """'
        test_iter: 10 test_interval: 10 base_lr: 0.01 momentum: 0.9
        weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
        display: 100 max_iter: 100 snapshot_after_train: false
        snapshot_prefix: "model" """)
    f.close()
    solver = caffe.SGDSolver(f.name)
    # also make sure get_solver runs
    caffe.get_solver(f.name)
    # fill in valid labels
    solver.net.blobs['label'].data[...] = np.random.randint(
        num_output, size=solver.net.blobs['label'].data.shape)
    solver.test_nets[0].blobs['label'].data[...] = np.random.randint(
        num_output, size=solver.test_nets[0].blobs['label'].data.shape)
    os.remove(f.name)
    os.remove(net_f)
    assert solver.iter == 0
    solver.solve()
    assert solver.iter == 100
