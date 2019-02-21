""" Hello world/smoke tests for important packages. """
import tempfile

# key packages in the anaconda meta-package


def test_bokeh():
    from bokeh.plotting import figure, output_file, show
    p = figure()
    p.circle([1, 2, 3], [4, 5, 6], color="orange")


def test_dask():
    import dask.array as da
    x = da.ones(10, chunks=(5,))
    assert x.sum() == 10


def test_h5py(tmpdir):
    import h5py
    fname = tmpdir.join('test.h5')
    f = h5py.File(fname, 'w')
    dset = f.create_dataset("mydataset", (100,), dtype='i')
    dset[:] = 9
    assert f['mydataset'][42] == 9


def test_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3])


def test_numba():
    from numba import jit
    import numpy as np

    @jit
    def sum(x, y):
        return x + y

    assert sum(1.0, 2.0) == 3.0

    @jit('f8(f8[:])')
    def sum1d(array):
        sum = 0.0
        for i in range(array.shape[0]):
            sum += array[i]
        return sum

    assert sum1d(np.array([1, 2, 3, 4], dtype='f8')) == 10


def test_numpy():
    import numpy
    a = numpy.arange(10)
    b = numpy.arange(10)
    c = a + b
    assert c[0] == 0
    assert c[1] == 2
    assert c[2] == 4


def test_pandas():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({
        'A': 1.,
        'B': pd.Timestamp('20130102'),
        'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        'D': np.array([3] * 4, dtype='int32'),
        'E': pd.Categorical(["test", "train", "test", "train"]),
        'F': 'foo'})
    df.dtypes
    df.head()
    df.tail()


def test_skimage():
    from skimage import data
    from skimage.color import rgb2gray
    original = data.astronaut()
    grayscale = rgb2gray(original)


def test_sklearn():
    # https://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.utils import check_random_state
    n = 100
    x = np.arange(n)
    rs = check_random_state(0)
    y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))
    ir = IsotonicRegression()
    y_ = ir.fit_transform(x, y)
    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)


def test_scipy():
    import numpy as np

    from scipy.ndimage import correlate
    result = correlate(np.arange(10), [1, 2.5])
    assert result[0] == 0
    assert result[-1] == 30

    from scipy import interpolate
    x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
    y = np.sin(x)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, 2*np.pi, np.pi/50)
    ynew = interpolate.splev(xnew, tck, der=0)


def test_statsmodels():
    import numpy as np
    import statsmodels.api as sm
    nsample = 100
    x = np.linspace(0, 10, 100)
    X = np.column_stack((x, x**2))
    beta = np.array([1, 0.1, 10])
    e = np.random.normal(size=nsample)
    X = sm.add_constant(X)
    y = np.dot(X, beta) + e
    model = sm.OLS(y, X)
    results = model.fit()


def test_sympy():
    from sympy import Rational
    a = Rational(1, 2)

    from sympy import Symbol
    x = Symbol('x')
    y = Symbol('y')
    ((x+y)**2).expand()


# engility requested packages


def test_mpi4py():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print("Hello! I'm rank %d from %d running in total..." %
          (comm.rank, comm.size))
    comm.Barrier()


def test_pydotplus():
    import pydotplus
    graph = pydotplus.Dot(graph_type='graph')
    edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 5)]
    nodes = [(1, "A"), (2, "B"), (3, "C"), (4, "D"), (5, "E")]
    for e in edges:
        graph.add_edge(pydotplus.Edge(e[0], e[1]))
    for n in nodes:
        node = pydotplus.Node(name=n[0], label=n[1])
        graph.add_node(node)


def test_altair():
    import altair as alt
    import pandas as pd
    source = pd.DataFrame({
        'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
        'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
    })
    alt.Chart(source).mark_bar().encode(x='a', y='b')


def test_hdbscan():
    from sklearn.datasets import make_blobs
    import hdbscan
    blobs, labels = make_blobs(n_samples=2000, n_features=10)
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(blobs)
    clusterer.labels_


def test_pymatgen():
    import pymatgen as mg
    si = mg.Element("Si")
    assert int(si.atomic_mass) == 28
    si.melting_point
    comp = mg.Composition("Fe2O3")
    assert int(comp.weight) == 159
    lattice = mg.Lattice.cubic(4.2)
    structure = mg.Structure(
        lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert int(structure.volume) == 74


def test_seaborn():
    import seaborn as sns
    sns.set(style="ticks")

    # Load the example dataset for Anscombe's quartet
    df = sns.load_dataset("anscombe")

    # Show the results of a linear regression within each dataset
    sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
               col_wrap=2, ci=None, palette="muted", height=4,
               scatter_kws={"s": 50, "alpha": 1})


# Deep learning frameworks

def test_tensorflow():
    import tensorflow as tf
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    out = sess.run(hello)
    a = tf.constant(10)
    b = tf.constant(32)
    sess.run(a+b)


def test_keras():
    # from keras test suite
    # https://github.com/keras-team/keras/blob/master/tests/integration_tests/test_vector_data_tasks.py
    # Classify random float vectors into 2 classes with logistic regression
    # using 2 layer neural network with ReLU hidden units.
    from keras.utils.test_utils import get_test_data
    from keras.models import Sequential
    from keras import layers
    import keras
    from keras.utils.np_utils import to_categorical
    num_classes = 2
    (x_train, y_train), (x_test, y_test) = get_test_data(
        num_train=500, num_test=200, input_shape=(20,), classification=True,
        num_classes=num_classes)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Test with Sequential API
    model = Sequential([
        layers.Dense(16, input_shape=(x_train.shape[-1],), activation='relu'),
        layers.Dense(8),
        layers.Activation('relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=15, batch_size=16,
                        validation_data=(x_test, y_test),
                        verbose=0)
    assert(history.history['val_acc'][-1] > 0.8)
    config = model.get_config()


def test_caffe():
    # adapted from caffe's test suite
    # https://github.com/BVLC/caffe/blob/master/python/caffe/test/test_solver.py
    import os

    import numpy as np

    import caffe

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
    caffe.set_mode_cpu()
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
