def test_pytorch():
    import torch

    # cpu test
    a = torch.ones(5)
    b = torch.ones(5)
    c = a + b
    assert c[0] == 2

    x = torch.ones(5)
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    assert z[0] == 2
    z.to("cpu", torch.double)
