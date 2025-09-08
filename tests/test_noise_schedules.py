import torch


def test_sigma_shapes_and_range():
    from models.noise_schedules import ddpm_like_sigma_from_tuni, edm_sigma_from_tuni

    t = torch.rand(4)
    sv = edm_sigma_from_tuni(t, P_mean=0.0, P_std=1.0, sigma_min=1e-3, sigma_max=1.0)
    sa = ddpm_like_sigma_from_tuni(t, sigma_min=1e-3, sigma_max=1.0)
    assert sv.shape == t.shape and sa.shape == t.shape
    assert torch.all(sv >= 1e-3) and torch.all(sv <= 1.0)
    assert torch.all(sa >= 1e-3) and torch.all(sa <= 1.0)
