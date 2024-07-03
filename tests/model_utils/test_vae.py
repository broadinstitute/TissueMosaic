import pytest
from tissuemosaic.model_utils.vae import VaeModel


@pytest.mark.parametrize("vae_type", ['vanilla', 'resnet18'])
def test_vae_construction(dummy_dino_dm, vae_type, trainer_no_logger, capsys):
    # construction Vae with params compatible with dummy_dm
    config = VaeModel.get_default_params()
    config['vae_type'] = vae_type
    config["global_size"] = dummy_dino_dm.global_size
    config["image_in_ch"] = dummy_dino_dm.ch_in
    vae_model = VaeModel(**config)
    assert isinstance(vae_model, VaeModel)
    # with capsys.disabled():
    #     print(vae)

    trainer_no_logger.fit(model=vae_model, datamodule=dummy_dino_dm)