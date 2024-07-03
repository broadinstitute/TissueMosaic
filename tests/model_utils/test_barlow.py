from tissuemosaic.model_utils.barlow import BarlowModel


def test_barlow_construction(dummy_dino_dm, trainer_no_logger, capsys):
    # construction Barlow with params compatible with dummy_dm
    config = BarlowModel.get_default_params()
    config["image_size"] = dummy_dino_dm.global_size
    config["image_in_ch"] = dummy_dino_dm.ch_in
    barlow = BarlowModel(**config)
    assert isinstance(barlow, BarlowModel)
    # with capsys.disabled():
    #     print(vae)

    trainer_no_logger.fit(model=barlow, datamodule=dummy_dino_dm)