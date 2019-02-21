def test_theano():
    import theano
    import theano.tensor as T
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    logistic = theano.function([x], s)
    results = logistic([[0, 1], [-1, -2]])
    assert 0.4 < results[0, 0] < 0.6
